#!/usr/bin/env python3
"""Train E023 references and freeze test predictions before physical evaluation."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
SEEDS = (23, 46, 69)
VIEWS = ("clean", "single", "mean4", "mean16")
TREE_GRID = tuple((leaf, fraction) for leaf in (1, 2, 4) for fraction in (0.5, 1.0))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(2, 128, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(129, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, trace, goal):
        _, hidden = self.gru(trace)
        return self.head(torch.cat((hidden[-1], goal[:, None]), 1))[:, 0]


def make_views(data):
    clean = data["trace_clean"].astype(np.float32)
    seed, sigma = int(data["noise_seed"]), float(data["noise_sigma"])
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, (len(clean), 16, clean.shape[1])).astype(np.float32)
    output = {"clean": clean}
    for name, count in (("single", 1), ("mean4", 4), ("mean16", 16)):
        view = clean.copy()
        view[:, :, 1] += noise[:, :count].mean(1)
        output[name] = view
    return output


def trace_stats(trace):
    mean = trace.mean((0, 1), keepdims=True).astype(np.float32)
    scale = (trace.std((0, 1), keepdims=True) + 1e-6).astype(np.float32)
    return mean, scale


def estimated_landing_error(prediction, data):
    return float(np.mean(np.abs(prediction - data["oracle_speed"]) * data["local_sensitivity"]))


def fit_gru(train_trace, validation_trace, train, validation, mean, scale, seed, epochs, device):
    torch.manual_seed(seed)
    x = ((train_trace - mean) / scale).astype(np.float32)
    vx = ((validation_trace - mean) / scale).astype(np.float32)
    goal = train["goal"].astype(np.float32)
    target_mean = np.float32(train["oracle_speed"].mean())
    target_scale = np.float32(train["oracle_speed"].std() + 1e-6)
    target = ((train["oracle_speed"] - target_mean) / target_scale).astype(np.float32)
    sensitivity_cap = np.float32(np.quantile(train["local_sensitivity"], 0.95))
    weight = np.minimum(train["local_sensitivity"], sensitivity_cap).astype(np.float32)
    weight /= np.float32(weight.mean() + 1e-6)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(goal),
                                      torch.from_numpy(target), torch.from_numpy(weight)),
                        batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model = Encoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    vx_t = torch.from_numpy(vx).to(device)
    vg_t = torch.from_numpy(validation["goal"].astype(np.float32)).to(device)
    best, best_metric, best_epoch = None, float("inf"), -1
    for epoch in range(epochs):
        model.train()
        for bx, bg, by, bw in loader:
            optimizer.zero_grad()
            error = model(bx.to(device), bg.to(device)) - by.to(device)
            loss = (bw.to(device) * error.square()).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(vx_t, vg_t).cpu().numpy() * target_scale + target_mean
        metric = estimated_landing_error(prediction, validation)
        if metric < best_metric:
            best_metric, best_epoch = metric, epoch + 1
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best)
    return model, {"selected_epoch": best_epoch,
                   "validation_estimated_landing_error_m": best_metric,
                   "target_mean": float(target_mean), "target_scale": float(target_scale),
                   "sensitivity_cap": float(sensitivity_cap)}


def predict_gru(model, trace, goal, mean, scale, target_mean, target_scale, device):
    x = ((trace - mean) / scale).astype(np.float32)
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(goal), 256):
            output.append(model(torch.from_numpy(x[start:start + 256]).to(device),
                                torch.from_numpy(goal[start:start + 256].astype(np.float32)).to(device)).cpu().numpy())
    return np.clip(np.concatenate(output) * target_scale + target_mean, 0, 3).astype(np.float32)


def flat(trace, goal):
    return np.concatenate((trace.reshape(len(trace), -1), goal[:, None]), axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(HERE / "data"))
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()
    root = Path(args.data)
    train, validation = np.load(root / "train.npz"), np.load(root / "validation.npz")
    train_views, validation_views = make_views(train), make_views(validation)
    mean, scale = trace_stats(train_views["clean"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoints = HERE / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    selection, models = {"gru": {}}, {}
    started = time.time()
    for view in VIEWS:
        selection["gru"][view], models[view] = [], []
        for seed in SEEDS:
            model, record = fit_gru(train_views[view], validation_views[view], train,
                                    validation, mean, scale, seed, args.epochs, device)
            models[view].append((model, record))
            selection["gru"][view].append({"seed": seed, **record})
            torch.save(model.state_dict(), checkpoints / f"gru_{view}_seed{seed}.pt")

    tree_trials = []
    clean_train, clean_validation = flat(train_views["clean"], train["goal"]), flat(validation_views["clean"], validation["goal"])
    for leaf, fraction in TREE_GRID:
        candidate = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=leaf,
                                        max_features=fraction, random_state=23023,
                                        n_jobs=-1).fit(clean_train, train["oracle_speed"])
        prediction = candidate.predict(clean_validation)
        tree_trials.append({"min_samples_leaf": leaf, "max_features": fraction,
                            "validation_estimated_landing_error_m": estimated_landing_error(prediction, validation)})
    selected_tree = min(tree_trials, key=lambda row: row["validation_estimated_landing_error_m"])
    trees, tree_validation = {}, {}
    for view in VIEWS:
        trees[view] = ExtraTreesRegressor(n_estimators=400,
            min_samples_leaf=selected_tree["min_samples_leaf"],
            max_features=selected_tree["max_features"], random_state=23023,
            n_jobs=-1).fit(flat(train_views[view], train["goal"]), train["oracle_speed"])
        tree_prediction = trees[view].predict(flat(validation_views[view], validation["goal"]))
        tree_validation[view] = estimated_landing_error(tree_prediction, validation)
    selection["tree_grid"] = tree_trials
    selection["selected_tree"] = selected_tree
    selection["tree_validation_estimated_landing_error_m"] = tree_validation
    clean_gru_predictions = []
    for model, record in models["clean"]:
        clean_gru_predictions.append(predict_gru(
            model, validation_views["clean"], validation["goal"], mean, scale,
            record["target_mean"], record["target_scale"], device
        ))
    clean_gru_validation = estimated_landing_error(
        np.mean(clean_gru_predictions, axis=0), validation
    )
    selection["clean_gru_ensemble_validation_estimated_landing_error_m"] = clean_gru_validation
    selection["primary_clean_method"] = "tree_clean" if tree_validation["clean"] < clean_gru_validation else "gru_clean_ensemble"

    np.savez_compressed(checkpoints / "preprocessing.npz", trace_mean=mean, trace_scale=scale)
    frozen_selection = {"status": "e023_selected_before_test", "device": device,
                        "elapsed_seconds": time.time() - started, "selection": selection}
    (HERE / "audit").mkdir(exist_ok=True)
    (HERE / "audit" / "selection.json").write_text(json.dumps(frozen_selection, indent=2) + "\n")

    test = np.load(root / "test.npz")
    test_views = make_views(test)
    predictions = {"oracle": test["oracle_speed"].astype(np.float32)}
    for view in VIEWS:
        seeds = []
        for seed, (model, record) in zip(SEEDS, models[view]):
            prediction = predict_gru(model, test_views[view], test["goal"], mean, scale,
                                     record["target_mean"], record["target_scale"], device)
            predictions[f"gru_{view}_seed{seed}"] = prediction
            seeds.append(prediction)
        predictions[f"gru_{view}_ensemble"] = np.mean(seeds, axis=0).astype(np.float32)
        predictions[f"tree_{view}"] = np.clip(trees[view].predict(flat(test_views[view], test["goal"])), 0, 3).astype(np.float32)
    np.savez_compressed(HERE / "audit" / "predictions_pre_evaluation.npz", **predictions)
    print(json.dumps(frozen_selection, indent=2))


if __name__ == "__main__":
    main()
