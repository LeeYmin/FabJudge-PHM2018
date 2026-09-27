"""06e correction of the 06d P2 RF field; original artifacts stay immutable."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .jev_gate import JEVClient, canonical, request_spec, validate_response, _state_float
from .jev_oof_edge import (FEATURES, auc, candidate_metrics, categories,
                           config_hash, grouped_bootstrap, logistic_oof,
                           predict_frozen, write_json)


def run(root: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    old = root / "artifacts/06d_lstm_jev_oof_edge"
    old_config_path = old / "frozen_config.json"
    old_config = json.loads(old_config_path.read_text(encoding="utf-8"))
    old_hash = config_hash(old_config_path)
    assert old_hash == (old / "frozen_config.sha256").read_text().strip()
    dev = pd.read_csv(old / "dev_rows.csv", float_precision="round_trip")
    test = pd.read_csv(old / "test_results.csv", float_precision="round_trip")
    question = old_config["questions"]["P2"]
    selected = old_config["sensor_features"]
    assert len(dev.loc[dev.lstm_alarm]) == 145 and len(test.loc[test.lstm_alarm]) == 23

    # Verify both the state and the actual 06d v2 cache record for all 23 test calls.
    test_cache_sha = []
    for _, row in test.loc[test.lstm_alarm].iterrows():
        spec = request_spec(row, question, selected, include_history=False,
                            serialization_version="v2")
        state_rf = spec["body"]["state"]["features"]["rf_probability"]
        expected = _state_float(row["rf_probability"], "v2")
        if state_rf != expected:
            raise RuntimeError("Stop: test P2 RF state differs from saved out-of-sample RF")
        sha = hashlib.sha256(canonical(spec["body"]).encode("utf-8")).hexdigest()
        path = old / "cache" / (sha + ".json")
        if not path.is_file():
            raise RuntimeError("Stop: missing test P2 cache request")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["request_sha256"] != sha:
            raise RuntimeError("Stop: test P2 cache request does not match saved RF")
        cached_score, _ = validate_response(record["response"], question)
        if cached_score != row["P2_score"]:
            raise RuntimeError("Stop: test P2 score differs from cached response")
        test_cache_sha.append(sha)

    client = JEVClient(root, out / "cache_06d_fix", 145, serialization_version="v2")
    alarm_indices = dev.index[dev.lstm_alarm]
    calls = []
    for ordinal, idx in enumerate(alarm_indices, start=1):
        row = dev.loc[idx]
        spec = request_spec(row, question, selected, include_history=False,
                            serialization_version="v2", rf_field="rf_probability_oof")
        if row.source == "train_oof":
            assert spec["body"]["state"]["features"]["rf_probability"] == _state_float(
                row["rf_probability_oof"], "v2")
        result = client.call_jev(row, question, selected, include_history=False,
                                 rf_field="rf_probability_oof")
        calls.append({"row_index": int(idx), "source": row.source,
                      "status": result["status"], "cache_hit": result["cache_hit"],
                      "score": result["score"], "error": result["error"]})
        if ordinal == 1 and result["score"] is None:
            pd.DataFrame(calls).to_csv(out / "call_log.csv", index=False)
            raise RuntimeError("Stop: first corrected JEV response schema failed")
        if result["score"] is None:
            pd.DataFrame(calls).to_csv(out / "call_log.csv", index=False)
            raise RuntimeError("Stop: unresolved corrected P2 response")
        dev.at[idx, "P2_score"] = result["score"]
        dev.at[idx, "P2_status"] = result["status"]
        if ordinal % 25 == 0:
            print(f"P2 corrected {ordinal}/145; new calls {client.new_calls}", flush=True)
    assert client.new_calls <= 145
    pd.DataFrame(calls).to_csv(out / "call_log.csv", index=False)
    dev.to_csv(out / "dev_rows_06d_fix.csv", index=False)

    # Preserve the exact preregistered candidate, threshold, and decision rules.
    config = json.loads(json.dumps(old_config))
    for key in ("phase", "seed", "rf_folds", "dev_sources", "candidates",
                "sensor_features", "sensor_rules", "questions", "jev_score_field",
                "logistic", "dev_evaluation", "bootstrap", "threshold_roles",
                "primary", "secondary", "combination_rule", "independence_rule",
                "dev_alarm_count", "dev_tp", "dev_fp"):
        assert config[key] == old_config[key]
    alarms = dev.loc[dev.lstm_alarm].copy().reset_index(drop=True)
    metrics = {}
    for name in ("J2", "S2"):
        if name == "J2":
            score = alarms.P2_score.to_numpy(float)
            model_info = None
        else:
            score, model = logistic_oof(alarms, FEATURES[name])
            scaler, classifier = model.steps[0][1], model.steps[1][1]
            model_info = {"features": FEATURES[name], "scaler_mean": scaler.mean_.tolist(),
                          "scaler_scale": scaler.scale_.tolist(), "coef": classifier.coef_[0].tolist(),
                          "intercept": float(classifier.intercept_[0])}
        alarms[f"{name}_oof_score"] = score
        m = candidate_metrics(alarms, name, f"{name}_oof_score", config["threshold_roles"])
        metrics[name] = m
        config["fitted"][name] = {"t_keep": m["t_keep"], "t_low": m["t_low"], "model": model_info}
        config["dev_candidate_auc"][name] = m["auc"]
    # B1 OOF values are unchanged; retain its original development scores.
    baseline = alarms.rf_probability_oof.to_numpy(float)
    alarms["B1_oof_score"] = baseline
    diff = lambda s: auc(s, "S2_oof_score") - auc(s, "B1_oof_score") if s.actual_fail_5000.nunique() == 2 else None
    config["auc_differences_vs_B1"]["S2"] = {
        "estimate": diff(alarms), "ci": grouped_bootstrap(alarms, diff)}
    frozen = out / "frozen_config_06d_fix.json"
    if frozen.exists() and json.loads(frozen.read_text(encoding="utf-8")) != config:
        raise RuntimeError("Stop: frozen 06d fix configuration mismatch")
    write_json(frozen, config)
    frozen_hash = config_hash(frozen)
    assert config_hash(frozen) == frozen_hash

    # One evaluation of the 06d test rows using only reused P2 responses.
    test_rows = test.loc[test.lstm_alarm].copy()
    comparisons = []
    old_comparison = pd.read_csv(old / "metric_comparison.csv")
    for name in ("B1", "B3", "S2", "S3", "J2", "J3"):
        if name not in {"S2", "J2"}:
            row = old_comparison.set_index("candidate").loc[name].to_dict()
            row.update({"candidate": name, "version": "06d_unchanged"})
            comparisons.append(row)
            continue
        score = predict_frozen(test_rows, name, config["fitted"][name], is_test=True)
        category = categories(score, config["fitted"][name]["t_keep"],
                              config["fitted"][name]["t_low"], config["threshold_roles"])
        truth = test_rows.actual_fail_5000.to_numpy(bool)
        lost = int(((category == "downgrade") & truth).sum())
        removed = int(((category == "downgrade") & ~truth).sum())
        frame = test_rows[["sequence_id", "actual_fail_5000"]].copy()
        frame["score"] = score
        ci = grouped_bootstrap(frame, lambda s: auc(s, "score"))
        full = test[["sequence_id", "actual_fail_5000"]].copy()
        full["category"] = "no_lstm_alarm"
        full.loc[test_rows.index, "category"] = category
        fp_ci = grouped_bootstrap(full, lambda s: int(((s.category == "downgrade") &
                                                       ~s.actual_fail_5000).sum()))
        recall_ci = grouped_bootstrap(full, lambda s: (float((s.category.isin(["keep", "review"]) &
                                s.actual_fail_5000).sum() / s.actual_fail_5000.sum())
                                if s.actual_fail_5000.sum() else None))
        row = old_comparison.set_index("candidate").loc[name].to_dict()
        row.update({"candidate": name, "version": "06e_corrected", "keep": int((category == "keep").sum()),
                    "review": int((category == "review").sum()),
                    "downgrade": int((category == "downgrade").sum()),
                    "TP": int(truth.sum()) - lost, "FP": int((~truth).sum()) - removed,
                    "FN": int(test.actual_fail_5000.sum()) - int(truth.sum()) + lost,
                    "TN": int((~test.actual_fail_5000).sum()) - int((~truth).sum()) + removed,
                    "alarm_tp_recall": (int(truth.sum()) - lost) / int(truth.sum()),
                    "lost_tp": lost, "removed_fp": removed, "constraint_met": lost <= 1,
                    "removed_fp_ci_low": fp_ci[0], "removed_fp_ci_high": fp_ci[1],
                    "recall_ci_low": recall_ci[0], "recall_ci_high": recall_ci[1],
                    "auc": auc(frame, "score"), "auc_ci_low": ci[0], "auc_ci_high": ci[1]})
        row["recall"] = row["TP"] / (row["TP"] + row["FN"])
        row["precision"] = row["TP"] / (row["TP"] + row["FP"]) if row["TP"] + row["FP"] else None
        comparisons.append(row)
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(out / "metric_comparison_06d_fix.csv", index=False)
    baseline_test = comparison.set_index("candidate").loc["B1"]
    s2_test = comparison.set_index("candidate").loc["S2"]
    s2_combination = bool(config["auc_differences_vs_B1"]["S2"]["ci"][0] > 0 and
                          s2_test.constraint_met and baseline_test.constraint_met and
                          s2_test.removed_fp >= baseline_test.removed_fp)
    original_dev = pd.read_csv(old / "dev_candidates.csv")
    rows = []
    for name in ("J2", "S2"):
        original = original_dev.loc[original_dev.candidate.eq(name) & original_dev.source.isna()].iloc[0]
        m = metrics[name]
        rows.append({"candidate": name, "original_dev_auc": float(original.auc),
                     "original_dev_auc_ci_low": float(original.auc_ci_low),
                     "original_dev_auc_ci_high": float(original.auc_ci_high),
                     "corrected_dev_auc": m["auc"], "corrected_dev_auc_ci_low": m["auc_ci_low"],
                     "corrected_dev_auc_ci_high": m["auc_ci_high"],
                     "original_rf_oof_spearman": float(original.spearman_rf),
                     "corrected_rf_oof_spearman": m["spearman_rf"],
                     "original_t_keep": float(original.t_keep), "original_t_low": float(original.t_low),
                     "corrected_t_keep": m["t_keep"], "corrected_t_low": m["t_low"]})
    pd.DataFrame(rows).to_csv(out / "dev_metric_comparison.csv", index=False)
    costs = []
    for path in (out / "cache_06d_fix").glob("*.json"):
        usage = json.loads(path.read_text(encoding="utf-8"))["response"].get("usage", {})
        if isinstance(usage.get("cost"), (int, float)):
            costs.append(float(usage["cost"]))
    response_count = len(list((out / "cache_06d_fix").glob("*.json")))
    summary = {"source_06d_config_sha256": old_hash, "frozen_config_sha256": frozen_hash,
               "test_state_verified": len(test_cache_sha), "test_cache_sha256": test_cache_sha,
               "new_jev_calls": response_count, "new_calls_this_execution": client.new_calls,
               "cache_hits": client.cache_hits,
               "actual_cost_usd": sum(costs), "estimated_cost_cap_usd": 145 * 2.5e-5,
               "s2_combination_effect_original": False,
               "s2_combination_effect_corrected": s2_combination,
               "s2_auc_difference_vs_b1": config["auc_differences_vs_B1"]["S2"]}
    write_json(out / "fix_summary.json", summary)
    return summary
