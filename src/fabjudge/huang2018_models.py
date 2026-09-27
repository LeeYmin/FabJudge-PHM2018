"""Leakage-safe Huang 2018 reproduction models and metrics.

The raw-window FabJudge feature branch is intentionally not imported here.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import random
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .huang2018 import (
    FAULT_NAMES, INPUT_COLUMNS, SHUTTER_CATEGORIES, SENSOR_COLUMNS,
    OPERATING_COLUMNS, SHUTTER_COLUMN, encode_inputs, sampled_raw_indices,
    smape_paper,
)


FAULT_SLUGS = ("fault1", "fault2", "fault3")
PAPER_METRICS = {
    "fault1": {"rfr_rmse_seconds": 5294, "lstm_rmse_seconds": 1877, "lstm_smape_percent": 13.90, "paper_train_sequences": 199, "paper_test_sequences": 39},
    "fault2": {"rfr_rmse_seconds": 5567, "lstm_rmse_seconds": 2557, "lstm_smape_percent": 16.94, "paper_train_sequences": 23, "paper_test_sequences": 4},
    "fault3": {"rfr_rmse_seconds": 5476, "lstm_rmse_seconds": 1469, "lstm_smape_percent": 11.74, "paper_train_sequences": 44, "paper_test_sequences": 10},
}


def make_original_splits(sequence_table: pd.DataFrame, output_path: Path, seed: int = 42) -> dict:
    """80/20 per equipment, with <4 failures kept for training; 90/10 validation."""
    rng = np.random.default_rng(seed)
    result = {}
    valid = sequence_table.loc[sequence_table["status"].eq("valid")].copy()
    for fault_name, slug in zip(FAULT_NAMES, FAULT_SLUGS):
        subset = valid.loc[valid["fault_name"].eq(fault_name)]
        train_pool = []
        test = []
        for _, equipment in subset.groupby("source_file", sort=True):
            ids = equipment["sequence_id"].to_numpy(dtype=str)
            ids = rng.permutation(ids).tolist()
            if len(ids) < 4:
                train_pool.extend(ids)
            else:
                n_test = max(1, round(0.2 * len(ids)))
                test.extend(ids[:n_test])
                train_pool.extend(ids[n_test:])
        train_pool = rng.permutation(train_pool).tolist()
        n_val = max(1, round(0.1 * len(train_pool)))
        val = train_pool[:n_val]
        train = train_pool[n_val:]
        assert set(train).isdisjoint(test)
        assert set(train).isdisjoint(val)
        assert set(val).isdisjoint(test)
        assert set(train) | set(val) | set(test) == set(subset["sequence_id"])
        result[slug] = {
            "fault_name": fault_name,
            "train_original_sequences": train,
            "validation_original_sequences": val,
            "test_original_sequences": test,
            "counts": {"train": len(train), "validation": len(val), "test": len(test)},
        }
    output_path.write_text(json.dumps({"seed": seed, "splits": result}, indent=2), encoding="utf-8")
    return result


def sequence_generator_unit_test() -> dict:
    expected = [np.array([90, 93, 96, 99]), np.array([89, 92, 95, 98]), np.array([88, 91, 94, 97])]
    observed = [sampled_raw_indices(99, max_length=4, sample_rate=3, offset=j) for j in range(3)]
    for actual, wanted in zip(observed, expected):
        np.testing.assert_array_equal(actual, wanted)
    np.testing.assert_array_equal(sampled_raw_indices(5, max_length=4, sample_rate=3), [2, 5])
    assert all(len(x) == 4 and np.all(np.diff(x) == 3) for x in observed)
    return {"synthetic_source_length": 100, "selected_indices_by_offset": [x.tolist() for x in observed], "short_history_indices": [2, 5], "passed": True}


def summarize_original(npz_path: Path, *, max_length: int = 300, sample_rate: int = 15, augmentations: int = 15) -> dict:
    with np.load(npz_path) as data:
        x_raw = data["x"]
        raw_indices = data["raw_row_index"]
        times = data["time"]
        y_raw = data["samples_to_fault"]
    n = len(x_raw)
    assert n > 3000 and x_raw.shape[1] == 21
    features = np.zeros((augmentations, max_length, 21), dtype=np.float32)
    targets = np.zeros((augmentations, max_length), dtype=np.float32)
    masks = np.zeros((augmentations, max_length), dtype=bool)
    source_indices = np.full((augmentations, max_length), -1, dtype=np.int64)
    source_times = np.full((augmentations, max_length), -1, dtype=np.int64)
    for offset in range(augmentations):
        selected = sampled_raw_indices(n - 1, max_length=max_length, sample_rate=sample_rate, offset=offset)
        length = len(selected)
        features[offset, -length:] = x_raw[selected]
        targets[offset, -length:] = y_raw[selected]
        masks[offset, -length:] = True
        source_indices[offset, -length:] = raw_indices[selected]
        source_times[offset, -length:] = times[selected]
        assert source_indices[offset, -1] == raw_indices[n - 1 - offset]
        assert np.all(np.diff(selected) == sample_rate)
    return {"x": features, "y": targets, "mask": masks, "raw_index": source_indices, "time": source_times}


def audit_position_only_reconstruction(project_root: Path, sequence_table: pd.DataFrame) -> dict:
    """Check how much target is determined by failure-anchored summary position."""
    max_error = 0.0
    checked = 0
    for _, row in sequence_table.loc[sequence_table["status"].eq("valid")].iterrows():
        summary = summarize_original(project_root / row["sequence_path"])
        for offset in range(15):
            length = int(summary["mask"][offset].sum())
            predicted = offset + 15 * np.arange(length - 1, -1, -1)
            actual = summary["y"][offset, summary["mask"][offset]]
            max_error = max(max_error, float(np.max(np.abs(predicted - actual))))
            checked += len(actual)
    return {
        "checked_sequences": int(sequence_table["status"].eq("valid").sum()),
        "checked_augmented_points": checked,
        "maximum_position_only_target_error_samples": max_error,
        "interpretation": "Known failure-anchor offset and within-summary position exactly determine this offline sample-count target; use causal rolling windows for online evaluation.",
    }


def load_summaries(project_root: Path, sequence_table: pd.DataFrame, ids: list[str]) -> dict:
    lookup = sequence_table.set_index("sequence_id")
    parts = []
    owners = []
    for sid in ids:
        row = lookup.loc[sid]
        path = project_root / row["sequence_path"]
        part = summarize_original(path)
        parts.append(part)
        owners.extend([sid] * len(part["x"]))
    if not parts:
        raise ValueError("Empty original sequence split")
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]} | {"owner": np.asarray(owners)}


class SequenceRUL(nn.Module):
    def __init__(self, input_size: int, h1: int, h2: int, dropout: float = 0.2):
        super().__init__()
        self.input_dropout = nn.Dropout(dropout)
        self.lstm1 = nn.LSTM(input_size, h1, batch_first=True)
        self.mid_dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(h1, h2, batch_first=True)
        self.head = nn.Sequential(nn.Linear(h2, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.lstm1(self.input_dropout(x))
        z, _ = self.lstm2(self.mid_dropout(z))
        return self.head(z).squeeze(-1)


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (((pred - target) ** 2) * mask).sum() / mask.sum().clamp(min=1)


def _loader(data: dict, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = TensorDataset(
        torch.from_numpy(data["x"]), torch.from_numpy(data["y"] / 4515.0),
        torch.from_numpy(data["mask"].astype(np.float32)),
    )
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _evaluate_loss(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    total = 0.0
    weight = 0.0
    with torch.no_grad():
        for x, y, mask in loader:
            pred = model(x)
            total += float((((pred - y) ** 2) * mask).sum())
            weight += float(mask.sum())
    return total / weight


def predict_lstm(model: nn.Module, data: dict, batch_size: int = 64) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(data["x"]), batch_size):
            predictions.append(model(torch.from_numpy(data["x"][start:start + batch_size])).numpy())
    return np.concatenate(predictions) * 4515.0


def intrinsic_metrics(data: dict, pred_samples: np.ndarray, seconds_per_sample: float = 4.0) -> dict:
    mask = data["mask"]
    actual = data["y"][mask].astype(np.float64) * seconds_per_sample
    predicted = pred_samples[mask].astype(np.float64) * seconds_per_sample
    return {
        "points": len(actual),
        "rmse_seconds": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "smape_percent": smape_paper(actual, predicted),
        "mae_seconds": float(np.mean(np.abs(predicted - actual))),
        "seconds_per_sample": seconds_per_sample,
    }


def train_repeated_lstm(
    slug: str, train_data: dict, val_data: dict, test_data: dict, model_dir: Path,
    *, repeats: int = 3, max_epochs: int = 40, patience: int = 6,
    batch_size: int = 64, seed: int = 42,
) -> tuple[SequenceRUL, dict]:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model_dir.mkdir(parents=True, exist_ok=True)
    h1, h2 = (64, 128) if slug in ("fault1", "fault2") else (32, 64)
    train_loader = _loader(train_data, batch_size, True)
    val_loader = _loader(val_data, batch_size, False)
    runs = []
    winner = None
    best_global = float("inf")
    for repetition in range(repeats):
        run_seed = seed + repetition
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        torch.use_deterministic_algorithms(True)
        model = SequenceRUL(21, h1, h2)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-5)
        best_val = float("inf")
        best_state = None
        best_epoch = 0
        best_train = float("nan")
        bad_epochs = 0
        started = time.perf_counter()
        for epoch in range(1, max_epochs + 1):
            model.train()
            train_sum = 0.0
            train_weight = 0.0
            for x, y, mask in train_loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x)
                loss = _masked_mse(prediction, y, mask)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite LSTM loss: {slug}, run {repetition}, epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_sum += float(loss) * float(mask.sum())
                train_weight += float(mask.sum())
            train_loss = train_sum / train_weight
            val_loss = _evaluate_loss(model, val_loader)
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_train = train_loss
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
            print(f"{slug} repeat {repetition+1}/{repeats} epoch {epoch}: train={train_loss:.6f} val={val_loss:.6f}", flush=True)
            if bad_epochs >= patience:
                break
        assert best_state is not None
        duration = time.perf_counter() - started
        runs.append({"repeat": repetition + 1, "seed": run_seed, "best_epoch": best_epoch, "epochs_run": epoch, "best_train_loss": best_train, "best_validation_loss": best_val, "duration_seconds": duration})
        if best_val < best_global:
            best_global = best_val
            winner = best_state
    assert winner is not None
    best_model = SequenceRUL(21, h1, h2)
    best_model.load_state_dict(winner)
    path = model_dir / f"{slug}_lstm.pt"
    torch.save({"state_dict": winner, "architecture": {"input_size": 21, "h1": h1, "h2": h2, "dropout": 0.2}, "target_scale_samples": 4515.0}, path)
    reloaded = torch.load(path, map_location="cpu", weights_only=True)
    verified = SequenceRUL(**reloaded["architecture"])
    verified.load_state_dict(reloaded["state_dict"])
    prediction = predict_lstm(verified, test_data)
    metrics = intrinsic_metrics(test_data, prediction)
    return verified, {"model_path": str(path), "runs": runs, "selected_repeat": int(np.argmin([x["best_validation_loss"] for x in runs])) + 1, "test_intrinsic": metrics}


def _rf_matrix(frame: pd.DataFrame, residual_model: RandomForestRegressor | None = None) -> np.ndarray:
    x = encode_inputs(frame)
    if residual_model is None:
        return x
    pressure = frame["FLOWCOOLPRESSURE"].to_numpy(dtype=np.float32).reshape(-1, 1)
    flow = frame["FLOWCOOLFLOWRATE"].to_numpy(dtype=np.float32)
    residual = flow - residual_model.predict(pressure).astype(np.float32)
    return np.column_stack((x, residual)).astype(np.float32)


def _rf_matrix_encoded(x: np.ndarray, residual_model: RandomForestRegressor) -> np.ndarray:
    pressure = x[:, SENSOR_COLUMNS.index("FLOWCOOLPRESSURE")].reshape(-1, 1)
    flow = x[:, SENSOR_COLUMNS.index("FLOWCOOLFLOWRATE")]
    residual = flow - residual_model.predict(pressure).astype(np.float32)
    return np.column_stack((x, residual)).astype(np.float32)


def load_rf_samples(artifact_dir: Path, fault_index: int) -> pd.DataFrame:
    frames = []
    for band in ("fault", "boundary", "normal"):
        path = artifact_dir / f"rf_sample_{fault_index}_{band}.parquet"
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def train_evaluate_rf(
    slug: str, fault_index: int, artifact_dir: Path, model_dir: Path,
    split: dict, *, seed: int = 42,
) -> tuple[dict, dict]:
    frame = load_rf_samples(artifact_dir, fault_index)
    held_out = set(split["test_original_sequences"]) | set(split["validation_original_sequences"])
    test_ids = set(split["test_original_sequences"])
    train = frame.loc[~frame["sequence_id"].isin(held_out) & frame["band"].isin(["fault", "normal"])].copy()
    test = frame.loc[frame["sequence_id"].isin(test_ids)].copy()
    normals = train.loc[train["band"].eq("normal")]
    faults = train.loc[train["band"].eq("fault")]
    if len(normals) == 0 or len(faults) == 0:
        raise ValueError(f"RF train lacks a class: {slug} normals={len(normals)}, faults={len(faults)}")
    pressure_col = "FLOWCOOLPRESSURE"
    flow_col = "FLOWCOOLFLOWRATE"
    relation = RandomForestRegressor(n_estimators=64, max_depth=10, min_samples_leaf=20, random_state=seed, n_jobs=4)
    relation.fit(normals[[pressure_col]].to_numpy(dtype=np.float32), normals[flow_col].to_numpy(dtype=np.float32))
    x_train = _rf_matrix(train, relation)
    y_train = train["band"].eq("fault").to_numpy(dtype=np.int8)
    # The paper duplicates every fault point 1000 times and draws equally many
    # normals. Here storage is Bernoulli sampled and per-row weights reproduce
    # the same balanced expected empirical risk, without identical RF bootstrap
    # draws. The difference is explicitly reported as a reproduction deviation.
    weights = np.where(y_train == 1, 1.0 / len(faults), 1.0 / len(normals)).astype(np.float64)
    weights *= len(train) / weights.sum()
    classifier = RandomForestClassifier(n_estimators=64, max_depth=16, min_samples_leaf=5, random_state=seed, n_jobs=4)
    started = time.perf_counter()
    classifier.fit(x_train, y_train, sample_weight=weights)
    duration = time.perf_counter() - started
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{slug}_rf.joblib"
    joblib.dump({"classifier": classifier, "residual_model": relation, "feature_names": [*INPUT_COLUMNS, "FLOWCOOLFLOWRATE_minus_fhat_FLOWCOOLPRESSURE"]}, path)
    reloaded = joblib.load(path)
    assert len(reloaded["feature_names"]) == 22
    probabilities = reloaded["classifier"].predict_proba(_rf_matrix(test, reloaded["residual_model"]))[:, 1]
    test = test.assign(probability=probabilities, prediction=probabilities >= 0.5)
    defined = test.loc[test["band"].isin(["fault", "normal"])]
    y = defined["band"].eq("fault").to_numpy(dtype=np.int8)
    p = defined["probability"].to_numpy(dtype=np.float64)
    predicted = p >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    # The stored test rows deliberately oversample faults. Recover population
    # metrics with inverse inclusion probabilities; recall/FAR are invariant
    # to class-only sampling, while precision, F1, and PR-AUC are not.
    inclusion = np.where(y == 1, 0.05, 0.0005)
    evaluation_weights = 1.0 / inclusion
    wtn, wfp, wfn, wtp = confusion_matrix(
        y, predicted, labels=[0, 1], sample_weight=evaluation_weights
    ).ravel()
    metrics = {
        "precision": float(precision_score(y, predicted, sample_weight=evaluation_weights, zero_division=0)),
        "recall": float(recall_score(y, predicted, sample_weight=evaluation_weights, zero_division=0)),
        "f1": float(f1_score(y, predicted, sample_weight=evaluation_weights, zero_division=0)),
        "confusion_matrix_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
        "estimated_population_confusion_tn_fp_fn_tp": [float(wtn), float(wfp), float(wfn), float(wtp)],
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, p, sample_weight=evaluation_weights)) if len(np.unique(y)) == 2 else None,
        "false_alarm_rate": float(fp / (fp + tn)) if fp + tn else None,
        "sampled_test_precision": float(precision_score(y, predicted, zero_division=0)),
        "sampled_test_f1": float(f1_score(y, predicted, zero_division=0)),
        "test_band_probability": {
            band: {
                "count": int(len(group)),
                "mean": float(group["probability"].mean()),
                "median": float(group["probability"].median()),
                "predicted_degrading_fraction": float(group["prediction"].mean()),
            }
            for band, group in test.groupby("band")
        },
        "stored_train_counts": {"fault": int(len(faults)), "normal": int(len(normals)), "boundary_excluded": int((frame["band"].eq("boundary") & ~frame["sequence_id"].isin(held_out)).sum())},
        "virtual_fault_count_after_x1000": int(len(faults) * 1000),
        "paper_virtual_normal_sample_count": int(len(faults) * 1000),
        "actual_weighted_rf_rows": int(len(train)),
        "duration_seconds": duration,
        "model_path": str(path),
        "test_sample_count": int(len(test)),
    }
    return reloaded, metrics


def combined_pipeline_metrics(
    rf_bundle: dict, lstm_model: SequenceRUL, test_data: dict,
    sequence_table: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    mask = test_data["mask"]
    x = test_data["x"][mask]
    owners = np.broadcast_to(test_data["owner"][:, None], mask.shape)[mask]
    times = test_data["time"][mask]
    raw_index = test_data["raw_index"][mask]
    actual_sample_seconds = test_data["y"][mask].astype(np.float64) * 4.0
    lookup = sequence_table.set_index("sequence_id")["fault_time"].to_dict()
    wall_seconds = np.asarray([lookup[sid] for sid in owners], dtype=np.float64) - times
    rf_probability = rf_bundle["classifier"].predict_proba(_rf_matrix_encoded(x, rf_bundle["residual_model"]))[:, 1]
    degrading = rf_probability >= 0.5
    lstm_seconds = predict_lstm(lstm_model, test_data)[mask].astype(np.float64) * 4.0
    pipeline_seconds = np.where(degrading, lstm_seconds, np.nan)
    fault = wall_seconds <= 5000
    normal = wall_seconds > 50000
    boundary = ~(fault | normal)
    rows = pd.DataFrame({
        "sequence_id": owners, "raw_row_index": raw_index,
        "time": times, "wall_ttf_seconds": wall_seconds,
        "sample_rul_seconds": actual_sample_seconds,
        "lstm_pred_seconds": lstm_seconds,
        "rf_probability": rf_probability,
        "pipeline_rul_seconds": pipeline_seconds,
        "band": np.where(fault, "fault", np.where(normal, "normal", "boundary")),
    }).drop_duplicates(["sequence_id", "raw_row_index"])
    detected_fault = rows.loc[rows["band"].eq("fault")]
    normal_rows = rows.loc[rows["band"].eq("normal")]
    detected = detected_fault.loc[detected_fault["pipeline_rul_seconds"].notna()]
    lead_times = []
    for _, group in rows.groupby("sequence_id"):
        positives = group.loc[
            group["pipeline_rul_seconds"].notna() & group["wall_ttf_seconds"].le(50000)
        ]
        if len(positives):
            lead_times.append(float(positives["wall_ttf_seconds"].max()))
    metrics = {
        "evaluated_tail_points": len(rows),
        "fault_detection_recall": float(len(detected) / len(detected_fault)) if len(detected_fault) else None,
        "false_alarm_rate_normal": float(normal_rows["pipeline_rul_seconds"].notna().mean()) if len(normal_rows) else None,
        "boundary_detection_fraction": float(rows.loc[rows["band"].eq("boundary"), "pipeline_rul_seconds"].notna().mean()) if (rows["band"] == "boundary").any() else None,
        "median_detection_lead_seconds_within_50000_band": float(np.median(lead_times)) if lead_times else None,
        "rul_rmse_after_detection_seconds": float(np.sqrt(np.mean((detected["pipeline_rul_seconds"] - detected["sample_rul_seconds"]) ** 2))) if len(detected) else None,
        "rul_smape_after_detection_percent": smape_paper(detected["sample_rul_seconds"].to_numpy(), detected["pipeline_rul_seconds"].to_numpy()) if len(detected) else None,
        "numerical_output_fraction": float(rows["pipeline_rul_seconds"].notna().mean()),
        "normal_output_nan_assertion": bool(rows.loc[(rows["band"] == "normal") & (rows["rf_probability"] < 0.5), "pipeline_rul_seconds"].isna().all()),
    }
    return metrics, rows


def combined_pipeline_metrics_causal(
    project_root: Path,
    rf_bundle: dict,
    lstm_model: SequenceRUL,
    sequence_table: pd.DataFrame,
    test_ids: list[str],
    *,
    endpoint_stride: int = 15,
    batch_size: int = 64,
) -> tuple[dict, pd.DataFrame]:
    """Deployable gate: every LSTM window ends at its current decision time.

    Paper's 15-offset failure-anchored summaries remain the intrinsic benchmark;
    this combined evaluation never constructs an input window using a future
    measurement or the known failure endpoint. Only labels use fault time.
    """
    lookup = sequence_table.set_index("sequence_id")
    parts = []
    lstm_model.eval()
    for sid in test_ids:
        row = lookup.loc[sid]
        with np.load(project_root / row["sequence_path"]) as data:
            raw_x = data["x"]
            raw_time = data["time"]
            raw_index = data["raw_row_index"]
            raw_target = data["samples_to_fault"]
        endpoints = np.unique(np.r_[np.arange(0, len(raw_x), endpoint_stride), len(raw_x)-1]).astype(np.int64)
        x_at_end = raw_x[endpoints]
        probabilities = rf_bundle["classifier"].predict_proba(
            _rf_matrix_encoded(x_at_end, rf_bundle["residual_model"])
        )[:, 1]
        predicted_samples = np.empty(len(endpoints), dtype=np.float64)
        for start in range(0, len(endpoints), batch_size):
            chosen = endpoints[start:start + batch_size]
            batch = np.zeros((len(chosen), 300, 21), dtype=np.float32)
            for j, endpoint in enumerate(chosen):
                indices = sampled_raw_indices(int(endpoint), max_length=300, sample_rate=15)
                assert np.all(indices <= endpoint)
                assert np.all(raw_time[indices] <= raw_time[endpoint])
                batch[j, -len(indices):] = raw_x[indices]
            with torch.no_grad():
                predicted_samples[start:start + len(chosen)] = (
                    lstm_model(torch.from_numpy(batch))[:, -1].numpy().astype(np.float64) * 4515.0
                )
        wall_seconds = int(row["fault_time"]) - raw_time[endpoints]
        fault = wall_seconds <= 5000
        normal = wall_seconds > 50000
        output = np.where(probabilities >= 0.5, predicted_samples * 4.0, np.nan)
        parts.append(pd.DataFrame({
            "sequence_id": sid,
            "raw_row_index": raw_index[endpoints],
            "time": raw_time[endpoints],
            "wall_ttf_seconds": wall_seconds,
            "sample_rul_seconds": raw_target[endpoints].astype(np.float64) * 4.0,
            "lstm_pred_seconds": predicted_samples * 4.0,
            "rf_probability": probabilities,
            "pipeline_rul_seconds": output,
            "band": np.where(fault, "fault", np.where(normal, "normal", "boundary")),
        }))
    rows = pd.concat(parts, ignore_index=True)
    fault_rows = rows.loc[rows["band"].eq("fault")]
    normal_rows = rows.loc[rows["band"].eq("normal")]
    boundary_rows = rows.loc[rows["band"].eq("boundary")]
    detected = fault_rows.loc[fault_rows["pipeline_rul_seconds"].notna()]
    lead_times = []
    for _, group in rows.groupby("sequence_id"):
        eligible = group.loc[
            group["wall_ttf_seconds"].le(50000) & group["pipeline_rul_seconds"].notna()
        ]
        if len(eligible):
            lead_times.append(float(eligible["wall_ttf_seconds"].max()))
    metrics = {
        "evaluation": "causal rolling endpoint every 15 raw measurements plus final endpoint",
        "evaluated_tail_points": int(len(rows)),
        "fault_detection_recall": float(len(detected) / len(fault_rows)) if len(fault_rows) else None,
        "false_alarm_rate_normal": float(normal_rows["pipeline_rul_seconds"].notna().mean()) if len(normal_rows) else None,
        "boundary_detection_fraction": float(boundary_rows["pipeline_rul_seconds"].notna().mean()) if len(boundary_rows) else None,
        "median_detection_lead_seconds_within_50000_band": float(np.median(lead_times)) if lead_times else None,
        "rul_rmse_after_detection_seconds": float(np.sqrt(np.mean((detected["pipeline_rul_seconds"] - detected["sample_rul_seconds"]) ** 2))) if len(detected) else None,
        "rul_smape_after_detection_percent": smape_paper(detected["sample_rul_seconds"].to_numpy(), detected["pipeline_rul_seconds"].to_numpy()) if len(detected) else None,
        "numerical_output_fraction": float(rows["pipeline_rul_seconds"].notna().mean()),
        "normal_output_nan_assertion": bool(rows.loc[rows["rf_probability"] < 0.5, "pipeline_rul_seconds"].isna().all()),
        "no_future_measurement_assertion": True,
    }
    return metrics, rows


def plot_test_curves(rows: pd.DataFrame, output_path: Path, *, n_sequences: int = 2) -> list[str]:
    import matplotlib.pyplot as plt

    ids = rows["sequence_id"].drop_duplicates().head(n_sequences).tolist()
    fig, axes = plt.subplots(len(ids), 2, figsize=(12, 4 * len(ids)), squeeze=False)
    for row_number, sid in enumerate(ids):
        group = rows.loc[rows["sequence_id"].eq(sid)].sort_values("sample_rul_seconds")
        x = group["sample_rul_seconds"].to_numpy() / 4.0
        actual = group["sample_rul_seconds"].to_numpy()
        pred = group["lstm_pred_seconds"].to_numpy()
        axes[row_number, 0].plot(x, actual, label="ground truth", linewidth=2)
        axes[row_number, 0].plot(x, pred, label="LSTM", alpha=0.8)
        axes[row_number, 0].set_ylabel("RUL (seconds)")
        axes[row_number, 0].set_xlabel("samples to fault")
        axes[row_number, 0].set_title(f"Test sequence {row_number+1}")
        axes[row_number, 0].legend()
        axes[row_number, 1].plot(x, np.abs(pred - actual), color="tab:red")
        axes[row_number, 1].set_ylabel("Absolute error (seconds)")
        axes[row_number, 1].set_xlabel("samples to fault")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return ids
