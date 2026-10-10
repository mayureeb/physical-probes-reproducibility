#!/usr/bin/env python3
"""Fit E024 baselines using train/validation only and freeze test predictions."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
SEEDS = (24, 48, 72)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)


def score(prediction: np.ndarray, target: np.ndarray, scale: np.ndarray) -> dict:
    normalized = (prediction - target) / scale
    return {"aggregate_nrmse": float(np.sqrt(np.mean(normalized ** 2))),
            "channel_rmse": np.sqrt(np.mean((prediction - target) ** 2, axis=0)).tolist()}


def wrong_history(history: np.ndarray, runs: np.ndarray) -> np.ndarray:
    output = history.copy()
    for run in np.unique(runs):
        indices = np.flatnonzero(runs == run)
        if len(indices) < 2:
            continue
        shift = max(1, len(indices) // 2)
        output[indices, :-1] = history[np.roll(indices, shift), :-1]
    return output


class HistoryGRU(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gru = nn.GRU(channels, 96, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(96, 96), nn.GELU(), nn.Linear(96, 3))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(history)
        return self.head(hidden[-1])


def predict_gru(model: nn.Module, history: np.ndarray, mean: np.ndarray,
                scale: np.ndarray, target_mean: np.ndarray,
                target_scale: np.ndarray, device: str) -> np.ndarray:
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(history), 1024):
            x = ((history[start:start + 1024] - mean) / scale).astype(np.float32)
            output.append(model(torch.from_numpy(x).to(device)).cpu().numpy())
    return np.concatenate(output) * target_scale + target_mean


def fit_gru(train: dict, validation: dict, mean: np.ndarray, feature_scale: np.ndarray,
            target_mean: np.ndarray, target_scale: np.ndarray, seed: int,
            epochs: int, device: str) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    x = ((train["history"] - mean) / feature_scale).astype(np.float32)
    y = ((train["target"] - target_mean) / target_scale).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=512, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model = HistoryGRU(x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    best_state, best_score, best_epoch = None, float("inf"), -1
    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = (model(bx.to(device)) - by.to(device)).square().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        prediction = predict_gru(model, validation["history"], mean, feature_scale,
                                 target_mean, target_scale, device)
        metric = score(prediction, validation["target"], target_scale)["aggregate_nrmse"]
        if metric < best_score:
            best_score, best_epoch = metric, epoch + 1
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, {"seed": seed, "selected_epoch": best_epoch,
                   "validation_aggregate_nrmse": best_score}


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def main() -> None:
    started = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    train, validation = load(args.data / "train.npz"), load(args.data / "validation.npz")
    feature_mean = train["history"].mean((0, 1), keepdims=True).astype(np.float32)
    feature_scale = (train["history"].std((0, 1), keepdims=True) + 1e-5).astype(np.float32)
    target_mean = train["target"].mean(0).astype(np.float32)
    target_scale = (train["target"].std(0) + 1e-5).astype(np.float32)
    x_current = (train["current"] - feature_mean.reshape(-1)[:6]) / feature_scale.reshape(-1)[:6]
    v_current = (validation["current"] - feature_mean.reshape(-1)[:6]) / feature_scale.reshape(-1)[:6]
    x_flat = ((train["history"] - feature_mean) / feature_scale).reshape(len(train["history"]), -1)
    v_flat = ((validation["history"] - feature_mean) / feature_scale).reshape(len(validation["history"]), -1)
    selection: dict[str, object] = {"status": "selected_without_test_targets",
                                    "constant_velocity_validation": score(
                                        validation["current"][:, :3], validation["target"],
                                        target_scale)}
    models: dict[str, object] = {}
    trials = []
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha).fit(x_current, train["target"])
        metric = score(model.predict(v_current), validation["target"], target_scale)
        trials.append({"alpha": alpha, **metric})
    chosen = min(trials, key=lambda item: item["aggregate_nrmse"])
    models["current_ridge"] = Ridge(alpha=chosen["alpha"]).fit(x_current, train["target"])
    selection["current_ridge"] = {"trials": trials, "selected": chosen}
    tree_trials = []
    for leaf in (4, 16, 64):
        model = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=leaf,
            max_features=1.0, n_jobs=-1, random_state=24024).fit(x_current, train["target"])
        tree_trials.append({"min_samples_leaf": leaf,
            **score(model.predict(v_current), validation["target"], target_scale)})
    chosen_tree = min(tree_trials, key=lambda item: item["aggregate_nrmse"])
    models["current_tree"] = ExtraTreesRegressor(n_estimators=320,
        min_samples_leaf=chosen_tree["min_samples_leaf"], max_features=1.0,
        n_jobs=-1, random_state=24024).fit(x_current, train["target"])
    selection["current_tree"] = {"trials": tree_trials, "selected": chosen_tree,
        "validation": score(models["current_tree"].predict(v_current), validation["target"], target_scale)}
    arx_trials = []
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha).fit(x_flat, train["target"])
        arx_trials.append({"alpha": alpha, **score(model.predict(v_flat), validation["target"], target_scale)})
    chosen_arx = min(arx_trials, key=lambda item: item["aggregate_nrmse"])
    models["arx"] = Ridge(alpha=chosen_arx["alpha"]).fit(x_flat, train["target"])
    selection["arx"] = {"trials": arx_trials, "selected": chosen_arx}
    rng = np.random.default_rng(24024)
    projection = rng.normal(0, 1 / np.sqrt(x_flat.shape[1]), (x_flat.shape[1], 512)).astype(np.float32)
    bias = rng.normal(0, 0.1, 512).astype(np.float32)
    xr = np.maximum(0, x_flat @ projection + bias)
    vr = np.maximum(0, v_flat @ projection + bias)
    random_trials = []
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha).fit(xr, train["target"])
        random_trials.append({"alpha": alpha, **score(model.predict(vr), validation["target"], target_scale)})
    chosen_random = min(random_trials, key=lambda item: item["aggregate_nrmse"])
    models["random_feature"] = Ridge(alpha=chosen_random["alpha"]).fit(xr, train["target"])
    selection["random_feature"] = {"trials": random_trials, "selected": chosen_random}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gru_models, gru_records, gru_predictions = [], [], []
    checkpoints = HERE / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    for seed in SEEDS:
        model, record = fit_gru(train, validation, feature_mean, feature_scale,
                                target_mean, target_scale, seed, args.epochs, device)
        torch.save(model.state_dict(), checkpoints / f"gru_seed{seed}.pt")
        gru_models.append(model)
        gru_records.append(record)
        gru_predictions.append(predict_gru(model, validation["history"], feature_mean,
                                            feature_scale, target_mean, target_scale, device))
    gru_validation = score(np.mean(gru_predictions, axis=0), validation["target"], target_scale)
    selection["gru"] = {"seeds": gru_records, "ensemble_validation": gru_validation}
    nonlinear = {"random_feature": chosen_random["aggregate_nrmse"],
                 "gru_ensemble": gru_validation["aggregate_nrmse"]}
    selection["primary_history_method"] = min(nonlinear, key=nonlinear.get)
    current_scores = {"current_ridge": chosen["aggregate_nrmse"],
                      "current_tree": selection["current_tree"]["validation"]["aggregate_nrmse"]}
    selection["primary_instant_method"] = min(current_scores, key=current_scores.get)
    joblib.dump(models, checkpoints / "classical.joblib")
    np.savez_compressed(checkpoints / "preprocessing.npz", feature_mean=feature_mean,
        feature_scale=feature_scale, target_mean=target_mean, target_scale=target_scale,
        random_projection=projection, random_bias=bias)
    selection["device"] = device
    selection["elapsed_seconds"] = time.time() - started
    selection["gru_parameter_count"] = sum(parameter.numel()
                                              for parameter in gru_models[0].parameters())
    (HERE / "audit").mkdir(exist_ok=True)
    (HERE / "audit" / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    predictions = {}
    for split in ("test_slag_heap", "test_damp"):
        test = load(args.data / f"{split}_inputs.npz")
        current = (test["current"] - feature_mean.reshape(-1)[:6]) / feature_scale.reshape(-1)[:6]
        flat = ((test["history"] - feature_mean) / feature_scale).reshape(len(test["history"]), -1)
        split_predictions = {
            "constant_velocity": test["current"][:, :3],
            "current_ridge": models["current_ridge"].predict(current),
            "current_tree": models["current_tree"].predict(current),
            "arx": models["arx"].predict(flat),
            "random_feature": models["random_feature"].predict(np.maximum(0, flat @ projection + bias)),
        }
        gru = [predict_gru(model, test["history"], feature_mean, feature_scale,
                           target_mean, target_scale, device) for model in gru_models]
        for seed, value in zip(SEEDS, gru):
            split_predictions[f"gru_seed{seed}"] = value
        split_predictions["gru_ensemble"] = np.mean(gru, axis=0)
        wrong = wrong_history(test["history"], test["run"])
        if selection["primary_history_method"] == "gru_ensemble":
            wrong_values = [predict_gru(model, wrong, feature_mean, feature_scale,
                                        target_mean, target_scale, device) for model in gru_models]
            split_predictions["primary_wrong_history"] = np.mean(wrong_values, axis=0)
        else:
            wrong_flat = ((wrong - feature_mean) / feature_scale).reshape(len(wrong), -1)
            split_predictions["primary_wrong_history"] = models["random_feature"].predict(
                np.maximum(0, wrong_flat @ projection + bias))
        predictions.update({f"{split}__{key}": value.astype(np.float32)
                            for key, value in split_predictions.items()})
    np.savez_compressed(HERE / "audit" / "predictions_pre_evaluation.npz", **predictions)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
