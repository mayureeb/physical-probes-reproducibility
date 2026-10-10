#!/usr/bin/env python3
"""Select E025 controls on calibration/validation and freeze test decisions."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
PROBES = ("passive", "fixed_positive_first", "half_then_full_forward",
          "fixed_negative_first", "multisine", "random_piecewise")
TRACE_INDEX = np.arange(0, 401, 50)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
GRU_SEEDS = (25, 50, 75)
VELOCITY_SIGMA = 0.01
FORCE_SIGMA = 0.04
FORCE_RHO = 0.8


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def seed_for(split: str, probe: str) -> int:
    return int(hashlib.sha256(f"E025:{split}:{probe}".encode()).hexdigest()[:8], 16)


def observed(data: dict, probe: str, split: str) -> np.ndarray:
    trace = data[f"{probe}__trace"][:, TRACE_INDEX].astype(np.float32).copy()
    rng = np.random.default_rng(seed_for(split, probe))
    trace[:, :, 0] += rng.normal(0, VELOCITY_SIGMA, trace[:, :, 0].shape)
    innovation = rng.normal(0, FORCE_SIGMA, trace[:, :, 1].shape).astype(np.float32)
    noise = np.empty_like(innovation)
    noise[:, 0] = innovation[:, 0]
    factor = np.sqrt(1 - FORCE_RHO ** 2)
    for index in range(1, noise.shape[1]):
        noise[:, index] = FORCE_RHO * noise[:, index - 1] + factor * innovation[:, index]
    trace[:, :, 1] += noise
    return trace


def flat(trace: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return np.concatenate((trace.reshape(len(trace), -1), goal[:, None]), axis=1)


def completion(prediction: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    return float(np.mean((prediction >= low) & (prediction <= high)))


def validation_score(prediction: np.ndarray, data: dict, probe: str) -> dict:
    target = data[f"{probe}__oracle_speed"]
    return {"completion": completion(prediction, data[f"{probe}__interval_low"],
                                      data[f"{probe}__interval_high"]),
            "speed_mae": float(np.mean(np.abs(prediction - target)))}


class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.GRU(3, 64, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        _, hidden = self.body(x)
        return self.head(hidden[-1]).squeeze(-1)


def sequence(trace: np.ndarray, goal: np.ndarray, mean: np.ndarray,
             scale: np.ndarray) -> np.ndarray:
    raw = np.concatenate((trace, np.repeat(goal[:, None, None], trace.shape[1], axis=1)), axis=2)
    return ((raw - mean) / scale).astype(np.float32)


def predict_gru(model: nn.Module, x: np.ndarray, target_mean: float,
                target_scale: float, device: str) -> np.ndarray:
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            values.append(model(torch.from_numpy(x[start:start + 1024]).to(device)).cpu().numpy())
    return np.concatenate(values) * target_scale + target_mean


def fit_gru(train_x: np.ndarray, train_y: np.ndarray, validation_x: np.ndarray,
            validation: dict, probe: str, seed: int, device: str):
    torch.manual_seed(seed)
    target_mean, target_scale = float(train_y.mean()), float(train_y.std() + 1e-5)
    y = ((train_y - target_mean) / target_scale).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(y)),
                        batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model = GRU().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    best = None
    for epoch in range(80):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = (model(bx.to(device)) - by.to(device)).square().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        prediction = predict_gru(model, validation_x, target_mean, target_scale, device)
        metric = validation_score(prediction, validation, probe)
        key = (-metric["completion"], metric["speed_mae"])
        if best is None or key < best[0]:
            best = (key, epoch + 1, {name: value.detach().cpu().clone()
                                    for name, value in model.state_dict().items()}, metric)
    model.load_state_dict(best[2])
    return model, target_mean, target_scale, {"seed": seed, "epoch": best[1], **best[3]}


def d_optimal_score(trace: np.ndarray, parameters: np.ndarray) -> float:
    x = trace.reshape(len(trace), -1).astype(np.float64)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    p = (parameters - parameters.mean(0)) / (parameters.std(0) + 1e-6)
    coefficient = np.linalg.lstsq(p, x, rcond=None)[0]
    sign, value = np.linalg.slogdet(coefficient @ coefficient.T + 1e-6 * np.eye(p.shape[1]))
    return float(value if sign > 0 else -np.inf)


def predict_selected(method: str, trace: np.ndarray, goal: np.ndarray, fitted: dict,
                     scaler: StandardScaler, projection: np.ndarray,
                     seq_mean: np.ndarray, seq_scale: np.ndarray,
                     gru_models: list, device: str) -> np.ndarray:
    values = flat(trace, goal)
    if method == "random_feature":
        return fitted[method].predict(np.maximum(0, scaler.transform(values) @ projection))
    if method == "gru_ensemble":
        x = sequence(trace, goal, seq_mean, seq_scale)
        return np.mean([predict_gru(model, x, target_mean, target_scale, device)
                        for model, target_mean, target_scale in gru_models], axis=0)
    return fitted[method].predict(values)


def main() -> None:
    started = time.time()
    data_root = HERE / "data"
    calibration, validation = load(data_root / "calibration.npz"), load(data_root / "validation.npz")
    test = load(data_root / "test_inputs.npz")
    fixed = "fixed_positive_first"
    caps = {"path_displacement": float(calibration[f"{fixed}__path_displacement"].max()),
            "absolute_work": float(calibration[f"{fixed}__absolute_work"].max())}
    costs, d_scores = {}, {}
    for probe in PROBES:
        path = calibration[f"{probe}__path_displacement"]
        work = calibration[f"{probe}__absolute_work"]
        costs[probe] = {"path_cap_violations": int(np.sum(path > caps["path_displacement"] + 1e-6)),
                        "work_cap_violations": int(np.sum(work > caps["absolute_work"] + 1e-6)),
                        "peak_speed_max": float(calibration[f"{probe}__peak_speed"].max())}
        if probe != "passive" and not costs[probe]["path_cap_violations"] and not costs[probe]["work_cap_violations"]:
            d_scores[probe] = d_optimal_score(calibration[f"{probe}__trace"][:, TRACE_INDEX],
                                              calibration["parameters"])
    if not d_scores:
        raise RuntimeError("no active probe satisfies inherited calibration caps")
    d_optimal = max(d_scores, key=d_scores.get)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoints = HERE / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    selection = {"status": "selected_without_test_targets", "caps": caps, "costs": costs,
                 "d_optimal_scores": d_scores, "d_optimal_probe": d_optimal,
                 "observation": {"velocity_sigma": VELOCITY_SIGMA,
                    "force_sigma": FORCE_SIGMA, "force_rho": FORCE_RHO,
                    "trace_indices": TRACE_INDEX.tolist()}, "probes": {}, "device": device}
    predictions = {}
    classical = {}
    for probe in PROBES:
        train_trace = observed(calibration, probe, "calibration")
        validation_trace = observed(validation, probe, "validation")
        test_trace = observed(test, probe, "test")
        train_x, validation_x, test_x = (flat(train_trace, calibration["goal"]),
            flat(validation_trace, validation["goal"]), flat(test_trace, test["goal"]))
        train_y = calibration[f"{probe}__oracle_speed"]
        trials, fitted = {}, {}
        for alpha in RIDGE_ALPHAS:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(train_x, train_y)
            trials[f"ridge_{alpha}"] = validation_score(model.predict(validation_x), validation, probe)
        alpha = min(RIDGE_ALPHAS, key=lambda a: (-trials[f"ridge_{a}"]["completion"],
                                                  trials[f"ridge_{a}"]["speed_mae"]))
        fitted["ridge"] = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(train_x, train_y)
        for leaf in (2, 4, 8, 16):
            model = ExtraTreesRegressor(n_estimators=256, min_samples_leaf=leaf,
                max_features=1.0, n_jobs=-1, random_state=25025).fit(train_x, train_y)
            trials[f"tree_{leaf}"] = validation_score(model.predict(validation_x), validation, probe)
        leaf = min((2, 4, 8, 16), key=lambda v: (-trials[f"tree_{v}"]["completion"],
                                                  trials[f"tree_{v}"]["speed_mae"]))
        fitted["tree"] = ExtraTreesRegressor(n_estimators=512, min_samples_leaf=leaf,
            max_features=1.0, n_jobs=-1, random_state=25025).fit(train_x, train_y)
        scaler = StandardScaler().fit(train_x)
        train_standard = scaler.transform(train_x)
        validation_standard = scaler.transform(validation_x)
        test_standard = scaler.transform(test_x)
        rng = np.random.default_rng(25025)
        projection = rng.normal(0, 1 / np.sqrt(train_x.shape[1]),
                                (train_x.shape[1], 256)).astype(np.float32)
        train_random = np.maximum(0, train_standard @ projection)
        validation_random = np.maximum(0, validation_standard @ projection)
        for alpha_random in RIDGE_ALPHAS:
            model = Ridge(alpha=alpha_random).fit(train_random, train_y)
            trials[f"random_{alpha_random}"] = validation_score(
                model.predict(validation_random), validation, probe)
        alpha_random = min(RIDGE_ALPHAS, key=lambda a: (-trials[f"random_{a}"]["completion"],
                                                         trials[f"random_{a}"]["speed_mae"]))
        fitted["random_feature"] = Ridge(alpha=alpha_random).fit(train_random, train_y)
        fitted["nearest"] = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=1)).fit(
            train_x, train_y)
        sequence_raw = np.concatenate((train_trace,
            np.repeat(calibration["goal"][:, None, None], train_trace.shape[1], axis=1)), axis=2)
        seq_mean = sequence_raw.mean((0, 1), keepdims=True).astype(np.float32)
        seq_scale = (sequence_raw.std((0, 1), keepdims=True) + 1e-5).astype(np.float32)
        train_seq = sequence(train_trace, calibration["goal"], seq_mean, seq_scale)
        validation_seq = sequence(validation_trace, validation["goal"], seq_mean, seq_scale)
        test_seq = sequence(test_trace, test["goal"], seq_mean, seq_scale)
        gru_models, gru_meta, gru_validation, gru_test = [], [], [], []
        for seed in GRU_SEEDS:
            model, target_mean, target_scale, record = fit_gru(
                train_seq, train_y, validation_seq, validation, probe, seed, device)
            torch.save({"state_dict": model.state_dict(), "target_mean": target_mean,
                        "target_scale": target_scale, "seq_mean": seq_mean,
                        "seq_scale": seq_scale}, checkpoints / f"{probe}_gru_seed{seed}.pt")
            gru_models.append((model, target_mean, target_scale))
            gru_meta.append(record)
            gru_validation.append(predict_gru(model, validation_seq, target_mean, target_scale, device))
            gru_test.append(predict_gru(model, test_seq, target_mean, target_scale, device))
        val_predictions = {
            "ridge": fitted["ridge"].predict(validation_x),
            "tree": fitted["tree"].predict(validation_x),
            "random_feature": fitted["random_feature"].predict(validation_random),
            "nearest": fitted["nearest"].predict(validation_x),
            "gru_ensemble": np.mean(gru_validation, axis=0)}
        val_scores = {name: validation_score(value, validation, probe)
                      for name, value in val_predictions.items()}
        primary = min(val_scores, key=lambda name: (-val_scores[name]["completion"],
                                                     val_scores[name]["speed_mae"]))
        test_predictions = {"ridge": fitted["ridge"].predict(test_x),
            "tree": fitted["tree"].predict(test_x),
            "random_feature": fitted["random_feature"].predict(
                np.maximum(0, test_standard @ projection)),
            "nearest": fitted["nearest"].predict(test_x),
            "gru_ensemble": np.mean(gru_test, axis=0)}
        for name, value in test_predictions.items():
            predictions[f"{probe}__{name}"] = value.astype(np.float32)
        wrong = np.roll(test_trace, len(test_trace) // 2, axis=0)
        wrong[:, -1] = test_trace[:, -1]
        force_masked = test_trace.copy()
        force_masked[:, :, 1] = float(train_trace[:, :, 1].mean())
        velocity_masked = test_trace.copy()
        velocity_masked[:, :, 0] = float(train_trace[:, :, 0].mean())
        predictions[f"{probe}__primary_wrong_history"] = predict_selected(
            primary, wrong, test["goal"], fitted, scaler, projection,
            seq_mean, seq_scale, gru_models, device).astype(np.float32)
        predictions[f"{probe}__primary_force_masked"] = predict_selected(
            primary, force_masked, test["goal"], fitted, scaler, projection,
            seq_mean, seq_scale, gru_models, device).astype(np.float32)
        predictions[f"{probe}__primary_velocity_masked"] = predict_selected(
            primary, velocity_masked, test["goal"], fitted, scaler, projection,
            seq_mean, seq_scale, gru_models, device).astype(np.float32)
        classical[probe] = {"models": fitted, "scaler": scaler, "projection": projection,
                            "seq_mean": seq_mean, "seq_scale": seq_scale}
        selection["probes"][probe] = {"trials": trials, "gru_seeds": gru_meta,
            "validation": val_scores, "primary_method": primary,
            "ridge_alpha": alpha, "tree_leaf": leaf, "random_alpha": alpha_random}
    selection["elapsed_seconds"] = time.time() - started
    selection["gru_parameter_count"] = sum(p.numel() for p in gru_models[0][0].parameters())
    selection["d_optimal_alias"] = d_optimal
    (HERE / "audit").mkdir(exist_ok=True)
    (HERE / "audit" / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    np.savez_compressed(HERE / "audit" / "predictions_pre_evaluation.npz", **predictions)
    joblib.dump(classical, checkpoints / "classical.joblib")
    print(json.dumps(selection, indent=2))
    joblib.dump(classical, checkpoints / "classical.joblib")
    (HERE / "audit").mkdir(exist_ok=True)
    (HERE / "audit" / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    np.savez_compressed(HERE / "audit" / "predictions_pre_evaluation.npz", **predictions)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
