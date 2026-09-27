"""Causal endpoint diagnosis and preregistered long-window prescreen for 06e."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .huang2018 import sampled_raw_indices
from .huang2018_models import SequenceRUL, _rf_matrix_encoded, load_rf_samples
from .jev_oof_edge import grouped_bootstrap, train_fold_rf, write_json

SENSORS = ("FLOWCOOLPRESSURE", "FLOWCOOLFLOWRATE", "pressure_flow_residual")
LONG = [f"{sensor}__{feature}" for sensor in SENSORS for feature in
        ("mean_z_1k", "mean_z_10k", "slope_10k", "std_ratio_10k")]
EXISTING = ["lstm_pred_seconds", "rf_probability_oof"]


def preregister(out: Path):
    payload = {
        "version": "06e-v2", "seed": 42, "test_sequences_used": False,
        "endpoints": "04b every 15 source rows plus final source row; current and prior only",
        "history": "baseline = sequence first 20000 seconds; require elapsed >= 30000 seconds",
        "sensors": list(SENSORS), "scale": "04b model input values; train residual uses held-sequence OOF normal relation; validation uses saved train-fit normal relation",
        "windows_seconds": [1000, 10000], "features": LONG,
        "feature_definitions": {
            "mean_z_1k": "(past/current 1000s mean - first 20000s baseline mean)/baseline population std",
            "mean_z_10k": "(past/current 10000s mean - first 20000s baseline mean)/baseline population std",
            "slope_10k": "OLS slope against actual seconds over past/current 10000s times 1000/baseline population std",
            "std_ratio_10k": "past/current 10000s population std/baseline population std",
            "zero_baseline_std": "insufficient_history; no imputation"},
        "ttf_bins_seconds": [0, 5000, 10000, 20000, 50000, "above 50000"],
        "tasks": {
            "T1": "fault3 LSTM alarm rows: actual TTF <=5000 versus >5000",
            "T2": "fault3 LSTM pred >5000: actual TTF <=20000 versus >20000",
            "T3_M01_to_M02": "T2 definition; fit on M01 sequences, evaluate M02 sequences",
            "T3_M02_to_M01": "T2 definition; fit on M02 sequences, evaluate M01 sequences",
            "T4_fault1": "fault1 T2 definition if artifacts exist",
            "T4_fault2": "fault2 T2 definition if artifacts exist"},
        "input_sets": {"E": EXISTING, "L": LONG, "E+L": EXISTING + LONG},
        "models": {"logistic": "StandardScaler, LogisticRegression(C=1,max_iter=1000,random_state=42)",
                   "RF": "RandomForestClassifier(n_estimators=100,max_depth=4,random_state=42,n_jobs=4)"},
        "validation": "GroupKFold(5) OOF by original sequence, except T3 strict other-machine holdout",
        "eligibility": "exclude a task if positive or negative evaluation sequences <5; T3 also requires fit classes",
        "metrics": "AUC; 1000 original-sequence paired bootstrap 95% CI; E+L minus E CI; T2 false-alarm rate at recall 0.8",
        "selection": "First choose tasks with E+L minus E AUC CI lower >0, largest difference first. Else choose L AUC CI lower >0.6 and E AUC <0.75. Else no 06f task. Evaluate both models and choose best eligible difference; fix E+L logistic and RF as 06f baselines.",
        "train_lstm_note": "train LSTM predictions are in sample; validation has four fault3 sequences"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":")).encode("utf-8")).hexdigest()
    record = {"sha256": digest, "payload": payload}
    path = out / "prereg_06e.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != record:
        prior = out / "prereg_06e_v1_superseded.json"
        old = json.loads(path.read_text(encoding="utf-8"))
        if not prior.exists() or json.loads(prior.read_text(encoding="utf-8")) != old or \
                old.get("sha256") != "a6c5fa02aa3081fcc116e78d953312d344639d8cb09c97c43537215b2a9e3e8e":
            raise RuntimeError("Stop: prereg hash/content mismatch")
    write_json(path, record)
    return digest


def _window_features(times: np.ndarray, signals: np.ndarray, endpoints: np.ndarray):
    """Return 12 defined features, with NaN only for insufficient history."""
    start = int(times[0])
    baseline = signals[times <= start + 20000]
    if len(baseline) < 2:
        raise ValueError("Missing sequence baseline")
    mean = baseline.mean(axis=0)
    std = baseline.std(axis=0)
    values = np.full((len(endpoints), 12), np.nan, dtype=np.float64)
    sufficient = np.zeros(len(endpoints), dtype=bool)
    for j, endpoint in enumerate(endpoints):
        t = times[endpoint]
        if t - start < 30000 or (std <= 0).any():
            continue
        short_start = np.searchsorted(times, t - 1000, side="left")
        long_start = np.searchsorted(times, t - 10000, side="left")
        short = signals[short_start:endpoint + 1]
        long = signals[long_start:endpoint + 1]
        tt = times[long_start:endpoint + 1].astype(np.float64)
        if len(short) < 2 or len(long) < 2:
            continue
        centered = tt - tt.mean()
        denom = np.dot(centered, centered)
        if denom <= 0:
            continue
        slope = centered @ (long - long.mean(axis=0)) / denom
        values[j] = np.column_stack(((short.mean(axis=0) - mean) / std,
                                      (long.mean(axis=0) - mean) / std,
                                      slope * 1000 / std, long.std(axis=0) / std)).ravel()
        sufficient[j] = True
    return values, sufficient


def _infer_sequence(path: Path, model, rf_relation, columns):
    with np.load(path) as data:
        x, times, raw_index = data["x"], data["time"], data["raw_row_index"]
    assert x.shape[1] == len(columns) and np.all(np.diff(times) >= 0)
    endpoints = np.unique(np.r_[np.arange(0, len(x), 15), len(x) - 1]).astype(int)
    predictions = np.empty(len(endpoints), dtype=float)
    for start in range(0, len(endpoints), 64):
        positions = endpoints[start:start + 64]
        batch = np.zeros((len(positions), 300, 21), dtype=np.float32)
        for j, endpoint in enumerate(positions):
            indices = sampled_raw_indices(int(endpoint), max_length=300, sample_rate=15)
            assert indices.max() <= endpoint and times[indices].max() <= times[endpoint]
            batch[j, -len(indices):] = x[indices]
        with torch.no_grad():
            predictions[start:start + len(positions)] = model(torch.from_numpy(batch))[:, -1].numpy() * 4515 * 4
    pressure = x[:, columns.index("FLOWCOOLPRESSURE")].reshape(-1, 1)
    flow = x[:, columns.index("FLOWCOOLFLOWRATE")]
    residual = flow - rf_relation.predict(pressure)
    signals = np.column_stack((pressure[:, 0], flow, residual))
    long_features, sufficient = _window_features(times, signals, endpoints)
    return x[endpoints], times[endpoints], raw_index[endpoints], predictions, long_features, sufficient


def input_check(root: Path, out: Path, split: dict, meta: pd.DataFrame, metrics: dict):
    faults = [slug for slug in ("fault1", "fault2", "fault3") if
              (root / f"artifacts/models/huang2018/{slug}_lstm.pt").is_file() and
              (root / f"artifacts/models/huang2018/{slug}_rf.joblib").is_file()]
    sid = split["fault3"]["train_original_sequences"][0]
    path = root / meta.set_index("sequence_id").loc[sid, "sequence_path"]
    with np.load(path) as data:
        times = data["time"]
    deltas = np.diff(times)
    check = {"lstm_max_input_rows": 300, "lstm_sample_rate_source_rows": 15,
             "sample_interval_seconds_quantiles": np.quantile(deltas, [0, .25, .5, .75, 1]).tolist(),
             "lstm_nominal_span_seconds": 299 * 15 * float(np.median(deltas)),
             "available_fault_artifacts": faults,
             "lstm_training_seconds_by_fault": {slug: sum(run["duration_seconds"] for run in metrics["lstm"][slug]["runs"]) for slug in faults},
             "fault3_validation_sequences": len(split["fault3"]["validation_original_sequences"]),
             "window_overlap_note": "10k window lies inside nominal 04b 17.94k-second maximum LSTM span; features summarize it differently. 1k also overlaps."}
    write_json(out / "lstm_input_check.json", check)
    return check


def build_endpoints(root: Path, out: Path, split: dict, meta: pd.DataFrame, columns: list[str], faults: list[str]):
    meta = meta.set_index("sequence_id")
    torch.set_num_threads(2)
    chunks = []
    for fault_index, slug in enumerate(("fault1", "fault2", "fault3"), start=1):
        if slug not in faults:
            continue
        checkpoint = torch.load(root / f"artifacts/models/huang2018/{slug}_lstm.pt",
                                map_location="cpu", weights_only=True)
        model = SequenceRUL(**checkpoint["architecture"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        rf = joblib.load(root / f"artifacts/models/huang2018/{slug}_rf.joblib")
        train_ids = split[slug]["train_original_sequences"]
        validation_ids = split[slug]["validation_original_sequences"]
        for number, sid in enumerate(train_ids + validation_ids, start=1):
            source = "train_oof" if sid in train_ids else "validation"
            path = (root / meta.loc[sid, "sequence_path"]).resolve()
            assert path.is_relative_to((root / "artifacts/huang2018/sequences").resolve())
            x, t, raw, pred, long, sufficient = _infer_sequence(path, model, rf["residual_model"], columns)
            frame = pd.DataFrame(long, columns=LONG)
            frame.insert(0, "sequence_id", sid)
            frame["fault"] = slug
            frame["source"] = source
            frame["machine"] = "M01" if "_M01_" in sid else "M02"
            frame["time"] = t
            frame["raw_row_index"] = raw
            frame["wall_ttf_seconds"] = int(meta.loc[sid, "fault_time"]) - t
            frame["lstm_pred_seconds"] = pred
            frame["insufficient_history"] = ~sufficient
            # Retain current encoded x temporarily to score train RF OOF.
            for i in range(len(columns)):
                frame[f"_x{i}"] = x[:, i]
            chunks.append(frame)
            if number % 25 == 0 or number == len(train_ids) + len(validation_ids):
                print(f"{slug}: {number}/{len(train_ids)+len(validation_ids)} sequences", flush=True)
        del model, checkpoint
    all_rows = pd.concat(chunks, ignore_index=True)
    for fault_index, slug in enumerate(("fault1", "fault2", "fault3"), start=1):
        if slug not in faults:
            continue
        mask = all_rows.fault.eq(slug)
        part = all_rows.loc[mask]
        train = part.loc[part.source.eq("train_oof")]
        frame = load_rf_samples(root / "artifacts/huang2018", fault_index)
        probability = pd.Series(np.nan, index=part.index, dtype=float)
        splitter = GroupKFold(n_splits=6)
        for fold, (fit, held) in enumerate(splitter.split(train, groups=train.sequence_id), start=1):
            fit_ids = set(train.iloc[fit].sequence_id)
            held_rows = train.iloc[held]
            classifier, relation, _ = train_fold_rf(frame, fit_ids)
            xx = held_rows[[f"_x{i}" for i in range(len(columns))]].to_numpy(np.float32)
            probability.loc[held_rows.index] = classifier.predict_proba(_rf_matrix_encoded(xx, relation))[:, 1]
            # The long-window residual must use the same held-sequence-free normal relation.
            for sid, group in held_rows.groupby("sequence_id", sort=False):
                path = (root / meta.loc[sid, "sequence_path"]).resolve()
                with np.load(path) as data:
                    full_x, full_time, full_raw = data["x"], data["time"], data["raw_row_index"]
                endpoint = np.unique(np.r_[np.arange(0, len(full_x), 15), len(full_x) - 1]).astype(int)
                assert np.array_equal(full_raw[endpoint], group.raw_row_index.to_numpy())
                pressure = full_x[:, columns.index("FLOWCOOLPRESSURE")].reshape(-1, 1)
                flow = full_x[:, columns.index("FLOWCOOLFLOWRATE")]
                residual = flow - relation.predict(pressure)
                signals = np.column_stack((pressure[:, 0], flow, residual))
                long, sufficient = _window_features(full_time, signals, endpoint)
                residual_columns = LONG[8:12]
                all_rows.loc[group.index, residual_columns] = long[:, 8:12]
                all_rows.loc[group.index, "insufficient_history"] = ~sufficient
            print(f"{slug} RF OOF fold {fold}/6", flush=True)
        validation = part.loc[part.source.eq("validation")]
        if len(validation):
            rf = joblib.load(root / f"artifacts/models/huang2018/{slug}_rf.joblib")
            xx = validation[[f"_x{i}" for i in range(len(columns))]].to_numpy(np.float32)
            probability.loc[validation.index] = rf["classifier"].predict_proba(_rf_matrix_encoded(xx, rf["residual_model"]))[:, 1]
        assert probability.notna().all()
        all_rows.loc[part.index, "rf_probability_oof"] = probability
    all_rows = all_rows.drop(columns=[c for c in all_rows if c.startswith("_x")])
    assert (all_rows.wall_ttf_seconds >= 0).all()
    all_rows.to_parquet(out / "long_window_features.parquet", index=False)
    return all_rows


def weakness_map(rows: pd.DataFrame, out: Path):
    work = rows.copy()
    work["ttf_band"] = pd.cut(work.wall_ttf_seconds, [-1, 5000, 10000, 20000, 50000, np.inf],
                              labels=["0-5k", "5k-10k", "10k-20k", "20k-50k", ">50k"])
    work["error_seconds"] = work.lstm_pred_seconds - work.wall_ttf_seconds
    work["target_5k"] = work.wall_ttf_seconds <= 5000
    work["alarm_5k"] = work.lstm_pred_seconds <= 5000
    records = []
    for fault, f in work.groupby("fault"):
        for unit in ("row", "sequence"):
            for dimension in ("ttf_band", "machine", "source"):
                for label, subset in f.groupby(dimension, observed=True):
                    if unit == "sequence":
                        subset = subset.groupby("sequence_id", as_index=False).agg(
                            error_seconds=("error_seconds", "median"), target_5k=("target_5k", "max"),
                            alarm_5k=("alarm_5k", "mean"), rf_probability_oof=("rf_probability_oof", "mean"))
                    error = subset.error_seconds.to_numpy(float)
                    y = subset.target_5k.astype(int)
                    auc = float(roc_auc_score(y, subset.rf_probability_oof)) if y.nunique() == 2 else None
                    records.append({"fault": fault, "unit": unit, "dimension": dimension, "group": str(label),
                                    "n": len(subset), "lstm_error_median_seconds": float(np.median(error)),
                                    "lstm_error_q1_seconds": float(np.quantile(error, .25)),
                                    "lstm_error_q3_seconds": float(np.quantile(error, .75)),
                                    "early_prediction_fraction": float((error < 0).mean()),
                                    "late_prediction_fraction": float((error > 0).mean()),
                                    "alarm_rate_5k": float(subset.alarm_5k.mean()),
                                    "rf_auc_5k": auc, "phm_asymmetric_cost": None,
                                    "note": "train LSTM in-sample; optimistic. PHM cost not applied."})
    result = pd.DataFrame(records)
    result.to_csv(out / "weakness_map.csv", index=False)
    return result


def prescreen(rows: pd.DataFrame, out: Path):
    usable = rows.loc[~rows.insufficient_history].copy()
    assert usable[EXISTING + LONG].notna().all().all()
    definitions = {
        "T1": usable.fault.eq("fault3") & usable.lstm_pred_seconds.le(5000),
        "T2": usable.fault.eq("fault3") & usable.lstm_pred_seconds.gt(5000),
        "T3_M01_to_M02": usable.fault.eq("fault3") & usable.lstm_pred_seconds.gt(5000),
        "T3_M02_to_M01": usable.fault.eq("fault3") & usable.lstm_pred_seconds.gt(5000),
        "T4_fault1": usable.fault.eq("fault1") & usable.lstm_pred_seconds.gt(5000),
        "T4_fault2": usable.fault.eq("fault2") & usable.lstm_pred_seconds.gt(5000)}
    results = []
    exclusions = []
    for task, mask in definitions.items():
        frame = usable.loc[mask].copy().reset_index(drop=True)
        frame["target"] = frame.wall_ttf_seconds.le(5000 if task == "T1" else 20000).astype(int)
        if task.startswith("T3_"):
            fit_machine, eval_machine = ("M01", "M02") if task.endswith("M01_to_M02") else ("M02", "M01")
            fit = frame.loc[frame.machine.eq(fit_machine)]
            evaluate = frame.loc[frame.machine.eq(eval_machine)]
        else:
            fit = frame
            evaluate = frame
        counts = evaluate.groupby("target").sequence_id.nunique().to_dict()
        if counts.get(0, 0) < 5 or counts.get(1, 0) < 5 or fit.target.nunique() < 2:
            exclusions.append({"task": task, "negative_sequences": counts.get(0, 0),
                               "positive_sequences": counts.get(1, 0), "reason": "<5 sequences in a class or fit lacks class"})
            continue
        for model_name in ("logistic", "RF"):
            predictions = {}
            for feature_set, columns in {"E": EXISTING, "L": LONG, "E+L": EXISTING + LONG}.items():
                def make_model():
                    if model_name == "logistic":
                        return make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=42))
                    return RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=4)
                if task.startswith("T3_"):
                    model = make_model()
                    model.fit(fit[columns], fit.target)
                    score = model.predict_proba(evaluate[columns])[:, 1]
                else:
                    score = np.full(len(frame), np.nan)
                    groups = frame.sequence_id.to_numpy()
                    for fit_idx, held_idx in GroupKFold(n_splits=5).split(frame, frame.target, groups):
                        model = make_model()
                        model.fit(frame.iloc[fit_idx][columns], frame.target.iloc[fit_idx])
                        score[held_idx] = model.predict_proba(frame.iloc[held_idx][columns])[:, 1]
                    assert np.isfinite(score).all()
                predictions[feature_set] = score
            scored = evaluate[["sequence_id", "target"]].copy().reset_index(drop=True)
            for key, score in predictions.items():
                scored[key] = score
            def metric(sample, column):
                return float(roc_auc_score(sample.target, sample[column])) if sample.target.nunique() == 2 else None
            delta = lambda s: metric(s, "E+L") - metric(s, "E") if s.target.nunique() == 2 else None
            row = {"task": task, "model": model_name, "rows": len(evaluate),
                   "sequences": evaluate.sequence_id.nunique(), "positive_sequences": counts[1],
                   "negative_sequences": counts[0], "validation": "machine holdout" if task.startswith("T3_") else "GroupKFold(5) OOF",
                   "auc_diff": delta(scored)}
            ci = grouped_bootstrap(scored, delta)
            row["auc_diff_ci_low"], row["auc_diff_ci_high"] = ci
            for key in ("E", "L", "E+L"):
                row[f"{key}_auc"] = metric(scored, key)
                ci = grouped_bootstrap(scored, lambda s, col=key: metric(s, col))
                row[f"{key}_auc_ci_low"], row[f"{key}_auc_ci_high"] = ci
            if task == "T2":
                for key in ("E", "L", "E+L"):
                    positive = scored.loc[scored.target.eq(1), key].to_numpy()
                    threshold = np.quantile(positive, .2)
                    row[f"{key}_false_alarm_rate_at_recall_08"] = float((scored.loc[scored.target.eq(0), key] >= threshold).mean())
            row["note"] = "train LSTM in-sample; optimistic. T3 machine transfer uses other-machine evaluation."
            results.append(row)
        print(f"prescreen {task}: complete", flush=True)
    pd.DataFrame(exclusions).to_csv(out / "excluded_tasks.csv", index=False)
    result = pd.DataFrame(results, columns=["task", "model", "rows", "sequences",
        "positive_sequences", "negative_sequences", "validation", "auc_diff",
        "auc_diff_ci_low", "auc_diff_ci_high", "E_auc", "E_auc_ci_low", "E_auc_ci_high",
        "L_auc", "L_auc_ci_low", "L_auc_ci_high", "E+L_auc", "E+L_auc_ci_low",
        "E+L_auc_ci_high", "note"])
    result.to_csv(out / "prescreen_results.csv", index=False)
    eligible = result.loc[result.auc_diff_ci_low.gt(0)].sort_values("auc_diff", ascending=False)
    if len(eligible):
        winner = eligible.iloc[0]
        reason = "positive paired AUC difference CI lower bound"
    else:
        eligible = result.loc[result.L_auc_ci_low.gt(.6) & result.E_auc.lt(.75)].sort_values("L_auc", ascending=False)
        winner = eligible.iloc[0] if len(eligible) else None
        reason = "L CI lower >0.6 and E AUC <0.75" if winner is not None else "no confirmed additional long-window signal"
    if result.empty:
        reason = "all preregistered tasks excluded: fewer than five positive or negative evaluation sequences"
    selection = {"task": str(winner.task) if winner is not None else None,
                 "winning_model": str(winner.model) if winner is not None else None,
                 "reason": reason, "baseline_models": ["E+L logistic", "E+L RF"] if winner is not None else [],
                 "excluded_tasks": exclusions,
                 "test_used": False, "jev_calls_part2": 0}
    write_json(out / "task_selection.json", selection)
    return selection


def run(root: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    digest = preregister(out)
    split = json.loads((root / "artifacts/huang2018/split_metadata.json").read_text(encoding="utf-8"))["splits"]
    meta = pd.read_csv(root / "artifacts/huang2018/sequence_metadata.csv")
    metrics = json.loads((root / "artifacts/huang2018/metrics.json").read_text(encoding="utf-8"))
    columns = json.loads((root / "artifacts/huang2018/reproduction_config.json").read_text(encoding="utf-8"))["model_input_columns"]
    check = input_check(root, out, split, meta, metrics)
    feature_path = out / "long_window_features.parquet"
    rows = (pd.read_parquet(feature_path) if feature_path.exists() else
            build_endpoints(root, out, split, meta, columns, check["available_fault_artifacts"]))
    weakness_map(rows, out)
    if preregister(out) != digest:
        raise RuntimeError("Stop: prereg hash mismatch")
    selection = prescreen(rows, out)
    return {"prereg_sha256": digest, "endpoint_rows": len(rows),
            "insufficient_history_rows": int(rows.insufficient_history.sum()),
            "faults": check["available_fault_artifacts"], "selection": selection}
