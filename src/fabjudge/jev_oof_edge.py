"""Reproducible stages for the fault3 06d notebook.

Only saved causal features and current/past sequence measurements are read.
Labels come from independent event metadata, never from NPZ targets.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .huang2018_models import _rf_matrix, _rf_matrix_encoded, load_rf_samples
from .jev_gate import canonical


NAMES = ("B1", "B3", "S2", "S3", "J2", "J3")
FEATURES = {
    "B1": ["rf_probability_oof"],
    "B3": ["rf_probability_oof", "ROTATIONSPEED", "ROTATIONSPEED__rms"],
    "S2": ["rf_probability_oof", "P2_score"],
    "S3": ["rf_probability_oof", "P3_score"],
    "J2": ["P2_score"],
    "J3": ["P3_score"],
}


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log(out: Path, text: str):
    with (out / "trial_log.md").open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n\n")


def train_fold_rf(frame: pd.DataFrame, train_ids: set[str], seed: int = 42):
    train = frame.loc[frame.sequence_id.isin(train_ids) & frame.band.isin(["fault", "normal"])]
    normals, faults = train.loc[train.band.eq("normal")], train.loc[train.band.eq("fault")]
    if not len(normals) or not len(faults):
        raise ValueError("RF fold lacks a class")
    relation = RandomForestRegressor(n_estimators=64, max_depth=10, min_samples_leaf=20,
                                     random_state=seed, n_jobs=4)
    relation.fit(normals[["FLOWCOOLPRESSURE"]].to_numpy(np.float32),
                 normals.FLOWCOOLFLOWRATE.to_numpy(np.float32))
    x = _rf_matrix(train, relation)
    y = train.band.eq("fault").to_numpy(np.int8)
    weights = np.where(y == 1, 1 / len(faults), 1 / len(normals)).astype(np.float64)
    weights *= len(train) / weights.sum()
    classifier = RandomForestClassifier(n_estimators=64, max_depth=16, min_samples_leaf=5,
                                        random_state=seed, n_jobs=4)
    classifier.fit(x, y, sample_weight=weights)
    return classifier, relation, {"fault": len(faults), "normal": len(normals)}


def endpoint_matrix(root: Path, meta: pd.DataFrame, rows: pd.DataFrame):
    chunks = []
    for sid, group in rows.groupby("sequence_id", sort=False):
        path = (root / str(meta.loc[sid, "sequence_path"])).resolve()
        assert path.is_relative_to((root / "artifacts/huang2018/sequences").resolve())
        with np.load(path) as data:
            x = data["x"]
            raw_index = data["raw_row_index"]
        positions = np.searchsorted(raw_index, group.raw_row_index.to_numpy())
        assert (raw_index[positions] == group.raw_row_index.to_numpy()).all()
        chunks.append(pd.DataFrame(x[positions], index=group.index))
    return pd.concat(chunks).loc[rows.index].to_numpy(np.float32)


def stage1(root: Path, out: Path, split: dict, meta: pd.DataFrame, input_columns: list[str]):
    parity = json.loads((out / "rf_parity.json").read_text(encoding="utf-8"))
    if parity["first_20_max_abs_error"] > .05:
        raise RuntimeError("Stop: RF parity error exceeds 0.05")
    train = pd.read_csv(root / "artifacts/06c_lstm_jev_semantic_gate/dev_rows.csv",
                        float_precision="round_trip")
    train = train.drop(columns=[c for c in train if c.startswith(("P0_", "P1_", "P2_")) or c == "pipeline_alarm"])
    assert len(train) == 360 and set(train.sequence_id) == set(split["train_original_sequences"])
    train["source"] = "train_oof"
    meta_index = meta.set_index("sequence_id")
    frame = load_rf_samples(root / "artifacts/huang2018", 3)
    groups = train.sequence_id.to_numpy()
    x_alarm = endpoint_matrix(root, meta_index, train.loc[train.lstm_alarm])
    alarm_indices = train.index[train.lstm_alarm].to_numpy()
    oof = np.full(len(train), np.nan)
    fold_log = []
    splitter = GroupKFold(n_splits=6)
    for fold, (fit_idx, held_idx) in enumerate(splitter.split(train, groups=groups), start=1):
        fit_ids = set(groups[fit_idx]); held_ids = set(groups[held_idx])
        assert fit_ids.isdisjoint(held_ids)
        classifier, relation, counts = train_fold_rf(frame, fit_ids)
        positions = np.flatnonzero(np.isin(groups[alarm_indices], list(held_ids)))
        oof[alarm_indices[positions]] = classifier.predict_proba(
            _rf_matrix_encoded(x_alarm[positions], relation))[:, 1]
        fold_log.append({"fold": fold, "fit_sequences": len(fit_ids), "held_sequences": len(held_ids),
                         "held_alarm_rows": len(positions), "rf_train_counts": counts})
    assert np.isfinite(oof[alarm_indices]).all()
    train["rf_probability_oof"] = oof
    import torch
    from .huang2018_models import SequenceRUL
    from .jev_gate import infer_causal_sampled_rows
    checkpoint = torch.load(root / "artifacts/models/huang2018/fault3_lstm.pt",
                            map_location="cpu", weights_only=True)
    lstm = SequenceRUL(**checkpoint["architecture"])
    lstm.load_state_dict(checkpoint["state_dict"])
    lstm.eval(); torch.set_num_threads(2)
    rf = joblib.load(out / "models/fault3_rf.joblib")
    validation, _ = infer_causal_sampled_rows(root, meta, split["validation_original_sequences"],
                                              rf, lstm, input_columns)
    validation["actual_fail_5000"] = validation.wall_ttf_seconds.le(5000)
    validation["lstm_alarm"] = validation.lstm_pred_seconds.le(5000)
    validation["source"] = "validation"
    validation["rf_probability_oof"] = validation.rf_probability
    dev = pd.concat([train, validation], ignore_index=True)
    alarms = dev.loc[dev.lstm_alarm].copy().reset_index(drop=True)
    assert alarms.actual_fail_5000.sum() >= 20 and (~alarms.actual_fail_5000).sum() >= 20
    dev.to_csv(out / "dev_rows.csv", index=False)
    records = []
    for source, group in alarms.groupby("source"):
        for label, subset in group.groupby("actual_fail_5000"):
            for field in (["rf_probability_oof", "rf_probability"] if source == "train_oof" else ["rf_probability_oof"]):
                vals = subset[field]
                records.append({"source": source, "class": "TP" if label else "FP", "field": field,
                                "n": len(vals), "median": vals.median(), "q1": vals.quantile(.25), "q3": vals.quantile(.75)})
    calibration = pd.DataFrame(records)
    calibration.to_csv(out / "calibration_check.csv", index=False)
    log(out, "## 1. RF OOF\n\n" + json.dumps(fold_log, ensure_ascii=False) +
        "\n\nRF uses 21 current-row inputs plus a pressure-flow residual; no LSTM output. "
        "The LSTM alarm set and train-side LSTM history remain in sample. "
        f"Dev alarms: {len(alarms)}, TP {int(alarms.actual_fail_5000.sum())}, "
        f"FP {int((~alarms.actual_fail_5000).sum())}.")
    return dev, alarms, calibration, fold_log


def make_questions(previous: dict):
    p2 = previous["questions"]["P2"]
    p3 = json.loads(json.dumps(p2))
    sentence = "rf_probability: estimated probability of a fault within 5000 seconds. "
    assert p3["instructions"].count(sentence) == 1
    p3["instructions"] = p3["instructions"].replace(sentence, "")
    for i, criterion in enumerate(p3["criteria"]):
        assert "LSTM trend, RF signal, and sensor evidence" in criterion
        p3["criteria"][i] = criterion.replace("LSTM trend, RF signal, and sensor evidence",
                                              "LSTM trend and sensor evidence")
    restored = json.loads(json.dumps(p3))
    restored["instructions"] = restored["instructions"].replace(
        "recent_lstm_delta:", sentence + "recent_lstm_delta:")
    restored["criteria"] = [line.replace("LSTM trend and sensor evidence",
                                           "LSTM trend, RF signal, and sensor evidence")
                            for line in restored["criteria"]]
    assert restored == p2
    return p2, p3


def make_prereg(out: Path, previous: dict, alarms: pd.DataFrame, p2: dict, p3: dict,
                threshold_roles: str):
    if threshold_roles not in {"swapped", "original_review_overlap"}:
        raise ValueError("Unresolved overlapping three-stage threshold definitions")
    config = {
        "phase": "preregistered_before_jev", "seed": 42, "rf_folds": 6,
        "dev_sources": ["train_oof", "validation"], "candidates": {n: FEATURES[n] for n in NAMES},
        "sensor_features": previous["selected_features"], "sensor_rules": previous["sensor_rules"],
        "questions": {"P2": p2, "P3": p3}, "jev_score_field": "score",
        "logistic": {"scaler": "StandardScaler", "C": 1, "max_iter": 1000, "random_state": 42},
        "dev_evaluation": "GroupKFold(6) out-of-fold by original sequence",
        "bootstrap": {"unit": "original sequence", "replicates": 1000, "seed": 42, "interval": "percentile 95%"},
        "threshold_roles": threshold_roles,
        "primary": "FP downgraded subject to <=10% loss of LSTM TP; test <=1 lost TP",
        "secondary": "alarm TP-vs-FP AUC and precision@k; k=round(dev FP fraction * alarm count)",
        "combination_rule": "dev grouped-bootstrap lower 95% CI of S2 or S3 AUC minus B1 AUC > 0 AND test recall-constrained FP removals >= B1",
        "independence_rule": "dev grouped-bootstrap J3 AUC lower 95% CI > 0.6 AND Spearman(J3, RF OOF) < 0.6",
        "dev_alarm_count": len(alarms), "dev_tp": int(alarms.actual_fail_5000.sum()),
        "dev_fp": int((~alarms.actual_fail_5000).sum()),
    }
    path = out / "frozen_config.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        assert old["questions"] == config["questions"] and old["threshold_roles"] == threshold_roles
        return old, config_hash(path)
    write_json(path, config)
    log(out, "## 3. Preregistration\n\n" + json.dumps(config, ensure_ascii=False, indent=2))
    return config, config_hash(path)


def logistic_oof(frame: pd.DataFrame, features: list[str]):
    y = frame.actual_fail_5000.to_numpy(int)
    groups = frame.sequence_id.to_numpy()
    oof = np.full(len(frame), np.nan)
    for fit, held in GroupKFold(n_splits=6).split(frame, y, groups):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=42))
        model.fit(frame.iloc[fit][features].to_numpy(float), y[fit])
        oof[held] = model.predict_proba(frame.iloc[held][features].to_numpy(float))[:, 1]
    assert np.isfinite(oof).all()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=42))
    model.fit(frame[features].to_numpy(float), y)
    return oof, model


def grouped_bootstrap(frame: pd.DataFrame, statistic, *, n=1000, seed=42):
    groups = list(frame.groupby("sequence_id").indices.values())
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        chosen = rng.integers(len(groups), size=len(groups))
        indices = np.concatenate([groups[i] for i in chosen])
        value = statistic(frame.iloc[indices])
        if value is not None and np.isfinite(value):
            values.append(float(value))
    if not values:
        return [None, None]
    return [float(x) for x in np.quantile(values, [.025, .975])]


def auc(frame: pd.DataFrame, column: str):
    if frame.actual_fail_5000.nunique() < 2:
        return None
    return float(roc_auc_score(frame.actual_fail_5000, frame[column]))


def threshold(scores, truth, target_recall):
    scores = np.asarray(scores, float); truth = np.asarray(truth, bool)
    assert np.isfinite(scores).all() and truth.any()
    values = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
    feasible = [float(t) for t in values if (scores[truth] >= t).mean() >= target_recall - 1e-12]
    assert feasible
    return max(feasible)


def categories(scores, t_keep, t_low, roles):
    scores = np.asarray(scores, float)
    if roles == "swapped":
        assert t_low <= t_keep
        return np.where(scores < t_low, "downgrade", np.where(scores >= t_keep, "keep", "review"))
    # Original thresholds overlap. A score in the overlap is held for review.
    assert t_keep <= t_low
    return np.where(scores < t_keep, "downgrade", np.where(scores >= t_low, "keep", "review"))


def candidate_metrics(frame: pd.DataFrame, name: str, score_col: str, roles: str):
    truth = frame.actual_fail_5000.to_numpy(bool)
    score = frame[score_col].to_numpy(float)
    if roles == "swapped":
        t_low = threshold(score, truth, .98)
        t_keep = threshold(score, truth, .90)
    else:
        t_keep = threshold(score, truth, .98)
        t_low = threshold(score, truth, .90)
    category = categories(score, t_keep, t_low, roles)
    removed = category == "downgrade"
    k = round((~truth).mean() * len(frame))
    queue = truth[np.argsort(-score, kind="stable")[:k]]
    ci = grouped_bootstrap(frame, lambda sample: auc(sample, score_col))
    return {"candidate": name, "auc": auc(frame, score_col),
            "auc_ci_low": ci[0], "auc_ci_high": ci[1],
            "spearman_rf": float(spearmanr(score, frame.rf_probability_oof).statistic),
            "t_keep": t_keep, "t_low": t_low, "keep": int((category == "keep").sum()),
            "review": int((category == "review").sum()), "downgrade": int(removed.sum()),
            "lost_tp": int((removed & truth).sum()), "removed_fp": int((removed & ~truth).sum()),
            "precision_at_k": float(queue.mean()), "k": k}


def stage2_calls(root: Path, out: Path, dev: pd.DataFrame, test: pd.DataFrame,
                 p2: dict, p3: dict, max_new_calls: int):
    from .jev_gate import JEVClient
    questions = {"P2": p2, "P3": p3}
    alarms = dev.loc[dev.lstm_alarm]
    client = JEVClient(root, out / "cache", max_new_calls,
                       serialization_version="v2")
    selected = ["ROTATIONSPEED", "ROTATIONSPEED__rms"]
    all_records = []
    for name, question in questions.items():
        for ordinal, (idx, row) in enumerate(alarms.iterrows()):
            result = client.call_jev(row, question, selected, include_history=False,
                                     include_rf=(name == "P2"))
            all_records.append({"stage": "dev", "question": name, "row_index": int(idx),
                                "status": result["status"], "error": result["error"],
                                "score": result["score"], "cache_hit": result["cache_hit"]})
            dev.at[idx, f"{name}_score"] = result["score"]
            dev.at[idx, f"{name}_status"] = result["status"]
            if ordinal == 0 and result["score"] is None:
                write_json(out / "call_failures.json", all_records)
                raise RuntimeError("Stop: first JEV response schema failed")
            if ordinal % 25 == 0:
                print(f"dev {name}: {ordinal + 1}/{len(alarms)}, new calls {client.new_calls}", flush=True)
        dev.to_csv(out / "dev_rows.csv", index=False)
    if dev.loc[dev.lstm_alarm, ["P2_score", "P3_score"]].isna().any().any():
        write_json(out / "call_failures.json", all_records)
        raise RuntimeError("Stop: unresolved dev JEV row")
    stability_records = []
    for idx, row in alarms.head(5).iterrows():
        for repeat in range(3):
            result = client.call_jev(row, p3, selected, include_history=False,
                                     include_rf=False, repeat_index=repeat)
            stability_records.append({"dev_index": int(idx), "sequence_id": row.sequence_id,
                                      "raw_row_index": int(row.raw_row_index), "repeat_index": repeat,
                                      "score": result["score"], "status": result["status"],
                                      "error": result["error"]})
            all_records.append({"stage": "stability", "question": "P3", **stability_records[-1]})
    stability = pd.DataFrame(stability_records)
    stability.to_csv(out / "stability.csv", index=False)
    if stability.score.isna().any():
        raise RuntimeError("Stop: unresolved stability response")
    log(out, f"## 4. Dev JEV\n\nP2/P3 success for {len(alarms)} alarms; "
        f"stability {len(stability)} calls. Cumulative new calls {client.new_calls}; "
        f"cache hits {client.cache_hits}.\n\nStability score SD per row: "
        + json.dumps(stability.groupby("dev_index").score.std(ddof=1).to_dict()))
    return dev, stability, client, all_records


def stage4_freeze(out: Path, dev: pd.DataFrame, config: dict, prereg_hash: str):
    path = out / "frozen_config.json"
    assert config_hash(path) == prereg_hash
    if config.get("phase") == "frozen_after_dev_before_test":
        saved_hash = (out / "frozen_config.sha256").read_text(encoding="utf-8").strip()
        if saved_hash != prereg_hash:
            raise RuntimeError("Stop: frozen config hash mismatch on replay")
        return (dev.loc[dev.lstm_alarm].copy().reset_index(drop=True),
                pd.read_csv(out / "dev_candidates.csv"), config, saved_hash)
    alarms = dev.loc[dev.lstm_alarm].copy().reset_index(drop=True)
    fitted = {}; records = []
    for name in NAMES:
        if name in {"B1", "J2", "J3"}:
            score = alarms[FEATURES[name][0]].to_numpy(float)
            model_info = None
        else:
            score, model = logistic_oof(alarms, FEATURES[name])
            scaler, classifier = model.steps[0][1], model.steps[1][1]
            model_info = {"features": FEATURES[name], "scaler_mean": scaler.mean_.tolist(),
                          "scaler_scale": scaler.scale_.tolist(), "coef": classifier.coef_[0].tolist(),
                          "intercept": float(classifier.intercept_[0])}
        alarms[f"{name}_oof_score"] = score
        result = candidate_metrics(alarms, name, f"{name}_oof_score", config["threshold_roles"])
        records.append(result)
        fitted[name] = {"t_keep": result["t_keep"], "t_low": result["t_low"],
                        "model": model_info}
        cats = categories(score, result["t_keep"], result["t_low"], config["threshold_roles"])
        alarms[f"{name}_category"] = cats
        for source, subset in alarms.groupby("source"):
            subcats = subset[f"{name}_category"].to_numpy()
            truth = subset.actual_fail_5000.to_numpy(bool)
            records.append({"candidate": name, "source": source, "n": len(subset),
                            "keep": int((subcats == "keep").sum()),
                            "review": int((subcats == "review").sum()),
                            "downgrade": int((subcats == "downgrade").sum()),
                            "lost_tp": int(((subcats == "downgrade") & truth).sum()),
                            "removed_fp": int(((subcats == "downgrade") & ~truth).sum()),
                            "auc": auc(subset, f"{name}_oof_score")})
    all_results = pd.DataFrame(records)
    all_results.to_csv(out / "dev_candidates.csv", index=False)
    auc_diffs = {}
    for name in ("S2", "S3"):
        diff = lambda sample: (auc(sample, f"{name}_oof_score") - auc(sample, "B1_oof_score")
                               if sample.actual_fail_5000.nunique() == 2 else None)
        auc_diffs[name] = {"estimate": diff(alarms), "ci": grouped_bootstrap(alarms, diff)}
    config["phase"] = "frozen_after_dev_before_test"
    config["fitted"] = fitted
    config["auc_differences_vs_B1"] = auc_diffs
    config["dev_candidate_auc"] = {row["candidate"]: row["auc"] for row in records
                                   if "source" not in row}
    write_json(path, config)
    frozen_hash = config_hash(path)
    (out / "frozen_config.sha256").write_text(frozen_hash + "\n", encoding="utf-8")
    log(out, "## 4. Frozen dev results\n\n```text\n" + all_results.to_string(index=False) + "\n```" +
        "\n\nAUC differences versus B1: " + json.dumps(auc_diffs) +
        f"\n\nFrozen SHA-256: `{frozen_hash}`")
    return alarms, all_results, config, frozen_hash


def predict_frozen(frame: pd.DataFrame, name: str, spec: dict, *, is_test=False):
    if name == "B1":
        return frame["rf_probability" if is_test else "rf_probability_oof"].to_numpy(float)
    if name in {"J2", "J3"}:
        return frame[f"P{name[-1]}_score"].to_numpy(float)
    info = spec["model"]
    features = ["rf_probability" if is_test and x == "rf_probability_oof" else x
                for x in info["features"]]
    x = frame[features].to_numpy(float)
    scaled = (x - np.array(info["scaler_mean"])) / np.array(info["scaler_scale"])
    z = scaled @ np.array(info["coef"]) + info["intercept"]
    return 1 / (1 + np.exp(-z))


def stage5_test(root: Path, out: Path, test: pd.DataFrame, config: dict,
                frozen_hash: str, client, all_records):
    assert config_hash(out / "frozen_config.json") == frozen_hash
    assert (out / "frozen_config.sha256").read_text(encoding="utf-8").strip() == frozen_hash
    if (out / "test_results.csv").exists():
        saved = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        if saved["frozen_config_sha256"] != frozen_hash:
            raise RuntimeError("Stop: prior test used a different frozen config")
        return pd.read_csv(out / "metric_comparison.csv"), saved
    p2, p3 = config["questions"]["P2"], config["questions"]["P3"]
    selected = config["sensor_features"]
    alarms = test.loc[test.lstm_alarm]
    for name, question in (("P2", p2), ("P3", p3)):
        for ordinal, (idx, row) in enumerate(alarms.iterrows()):
            result = client.call_jev(row, question, selected, include_history=False,
                                     include_rf=(name == "P2"))
            all_records.append({"stage": "test", "question": name, "row_index": int(idx),
                                "status": result["status"], "error": result["error"],
                                "score": result["score"], "cache_hit": result["cache_hit"]})
            test.at[idx, f"{name}_score"] = result["score"]
            if ordinal == 0 and result["score"] is None:
                raise RuntimeError("Stop: first test JEV response invalid")
        if test.loc[test.lstm_alarm, f"{name}_score"].isna().any():
            raise RuntimeError("Stop: unresolved test JEV row")
    alarms = test.loc[test.lstm_alarm].copy()
    summary_rows = []; first_rows = []
    for name in NAMES:
        spec = config["fitted"][name]
        score = predict_frozen(alarms, name, spec, is_test=True)
        category = categories(score, spec["t_keep"], spec["t_low"], config["threshold_roles"])
        test[f"{name}_score"] = np.nan
        test.loc[alarms.index, f"{name}_score"] = score
        test[f"{name}_category"] = "no_lstm_alarm"
        test.loc[alarms.index, f"{name}_category"] = category
        pred = test[f"{name}_category"].isin(["keep", "review"]).to_numpy()
        truth = test.actual_fail_5000.to_numpy(bool)
        tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
        lost = int(((category == "downgrade") & alarms.actual_fail_5000.to_numpy(bool)).sum())
        removed = int(((category == "downgrade") & ~alarms.actual_fail_5000.to_numpy(bool)).sum())
        work = test[["sequence_id", "actual_fail_5000", "lstm_alarm", f"{name}_category"]].copy()
        work.columns = ["sequence_id", "actual_fail_5000", "lstm_alarm", "category"]
        ci_fp = grouped_bootstrap(work, lambda s: ((s.category == "downgrade") &
                                                   ~s.actual_fail_5000).sum())
        ci_recall = grouped_bootstrap(work, lambda s: (
            float((s.category.isin(["keep", "review"]) & s.actual_fail_5000).sum() /
                  s.actual_fail_5000.sum()) if s.actual_fail_5000.sum() else None))
        score_frame = alarms[["sequence_id", "actual_fail_5000"]].copy()
        score_frame["score"] = score
        auc_ci = grouped_bootstrap(score_frame, lambda s: auc(s, "score"))
        summary_rows.append({"candidate": name, "keep": int((category == "keep").sum()),
                             "review": int((category == "review").sum()),
                             "downgrade": int((category == "downgrade").sum()),
                             "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                             "recall": tp / (tp + fn), "precision": tp / (tp + fp) if tp + fp else None,
                             "alarm_tp_recall": (int(alarms.actual_fail_5000.sum()) - lost) /
                                                int(alarms.actual_fail_5000.sum()),
                             "lost_tp": lost, "removed_fp": removed,
                             "constraint_met": lost <= 1,
                             "removed_fp_ci_low": ci_fp[0], "removed_fp_ci_high": ci_fp[1],
                             "recall_ci_low": ci_recall[0], "recall_ci_high": ci_recall[1],
                             "auc": auc(score_frame, "score"), "auc_ci_low": auc_ci[0],
                             "auc_ci_high": auc_ci[1]})
        for sid, group in test.groupby("sequence_id", sort=False):
            first = group.loc[group[f"{name}_category"].isin(["keep", "review"])].sort_values("time").head(1)
            first_rows.append({"candidate": name, "sequence_id": sid,
                               "first_alarm_time": int(first.time.iloc[0]) if len(first) else None,
                               "first_alarm_raw_row_index": int(first.raw_row_index.iloc[0]) if len(first) else None})
    comparison = pd.DataFrame(summary_rows)
    prior = json.loads((root / "artifacts/06c_lstm_jev_semantic_gate/summary.json").read_text(encoding="utf-8"))
    reference = next(m for m in prior["metrics"] if m["system"] == "P2")
    comparison["06c_P2_reference_TP"] = reference["TP"]
    comparison["06c_P2_reference_FP"] = reference["FP"]
    comparison["06c_P2_reference_recall"] = reference["recall"]
    test.to_csv(out / "test_results.csv", index=False)
    comparison.to_csv(out / "metric_comparison.csv", index=False)
    pd.DataFrame(first_rows).to_csv(out / "first_alarm_by_sequence.csv", index=False)
    combination = {}
    for name in ("S2", "S3"):
        diff = config["auc_differences_vs_B1"][name]
        candidate = comparison.set_index("candidate").loc[name]
        baseline = comparison.set_index("candidate").loc["B1"]
        combination[name] = bool(diff["ci"][0] is not None and diff["ci"][0] > 0 and
                                 candidate.constraint_met and baseline.constraint_met and
                                 candidate.removed_fp >= baseline.removed_fp)
    dev = pd.read_csv(out / "dev_rows.csv")
    dev = dev.loc[dev.lstm_alarm]
    j3_table = pd.read_csv(out / "dev_candidates.csv")
    j3_ci = j3_table.loc[j3_table.candidate.eq("J3") & j3_table.source.isna()]
    j3_ci = j3_ci.iloc[0]
    correlation = float(spearmanr(dev.P3_score, dev.rf_probability_oof).statistic)
    independent = bool(j3_ci.auc_ci_low > .6 and correlation < .6)
    eligible = comparison.loc[comparison.constraint_met]
    maximum = int(eligible.removed_fp.max()) if len(eligible) else None
    winners = eligible.loc[eligible.removed_fp.eq(maximum)].candidate.tolist() if len(eligible) else []
    response_files = list((out / "cache").rglob("*.json"))
    summary = {"frozen_config_sha256": frozen_hash, "total_successful_calls": len(response_files),
               "new_calls_this_execution": client.new_calls,
               "cache_hits_this_execution": client.cache_hits, "max_new_calls": client.max_new_calls,
               "estimated_cost_usd": None, "actual_cost_usd": None,
               "dev_alarms": int(dev.shape[0]), "test_alarms": int(len(alarms)),
               "combination_effect": combination, "independent_signal": independent,
               "j3_rf_spearman": correlation, "eligible_max_removed_fp": maximum,
               "eligible_winners": winners,
               "limits": ["train LSTM output and alarm set are in sample", "test has 8 sequences",
                          "fault3 is one distribution"]}
    costs = []
    for path in response_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        usage = record["response"].get("usage", {})
        if isinstance(usage.get("cost"), (int, float)):
            costs.append(usage["cost"])
    summary["actual_cost_usd"] = float(sum(costs))
    summary["estimated_cost_usd"] = float(client.max_new_calls * 2.5e-5)
    write_json(out / "summary.json", summary)
    log(out, "## 5. One test evaluation\n\n```text\n" + comparison.to_string(index=False) + "\n```" +
        "\n\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    return comparison, summary
