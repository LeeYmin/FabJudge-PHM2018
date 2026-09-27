"""Observed 05b group application of the locked exploratory Noul rule."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from . import jev_noul_analysis as analysis
from .jev_noul_followup import (Client, _attach_local, _network_budget_allows_completion,
                                 load_context, read_json, read_key, reconstruct_features, write_json)


def run_evaluation(root: Path):
    ctx = load_context(Path(root).resolve())
    out, sample = ctx["out"], ctx["sample"]
    lock_path = out / "development_rule_lock.json"
    if not lock_path.is_file():
        raise RuntimeError("Development thresholds must be locked before evaluating observed 05b groups.")
    lock = read_json(lock_path)
    if not lock.get("locked_before_new_evaluation_requests") or lock.get("independent_holdout_claim") is not False:
        raise ValueError("Invalid development rule lock.")
    if read_json(out / "run_status.json").get("stage") != "development_complete_rules_locked":
        raise RuntimeError("Development stage is incomplete.")
    development = pd.read_csv(out / "development_responses.csv")
    if len(development) != 180 or not development.success.all():
        raise ValueError("Development responses are incomplete.")
    raw_rows = reconstruct_features(ctx)
    key = read_key(ctx["root"])
    if not key:
        raise RuntimeError("JEV_API_KEY or OPENROUTER_API_KEY is not configured.")
    client = Client(ctx, key)
    observed = (sample.loc[sample.subset == "holdout"]
                .sort_values(["target", "region", "sample_id"]).reset_index(drop=True))
    records = []
    for row in observed.itertuples(index=False):
        allowed, projection = _network_budget_allows_completion(client)
        if not allowed:
            write_json(out / "run_status.json", {"stage":"observed_evaluation_budget_stop",
                       "projection":projection, "usage":client.usage(), "independent_holdout":False})
            raise RuntimeError("Remaining planned requests no longer fit the new hard limits.")
        rec = client.invoke(row, raw_rows)
        records.append(_attach_local(rec, row))
        if not rec.get("success"):
            break
    frame = pd.DataFrame(records)
    frame.to_csv(out / "observed_05b_holdout_responses.csv", index=False)
    if len(frame) != 90 or not frame.success.all():
        write_json(out / "run_status.json", {"stage":"observed_evaluation_incomplete",
                   "evaluation_rows":len(frame), "usage":client.usage(), "independent_holdout":False})
        raise RuntimeError("Observed-group evaluation responses are incomplete or invalid.")
    if frame.group_id.nunique() != 57 or set(frame.group_id) & set(development.group_id):
        raise ValueError("Observed-group separation differs from the audited plan.")
    assigned = analysis.assign_predictions(frame, lock["thresholds"])
    assigned.to_csv(out / "observed_05b_holdout_predictions.csv", index=False)
    metric, confusion, classes = analysis.three_class_tables(assigned, "previously_observed_05b_holdout")
    metric.to_csv(out / "observed_05b_holdout_metrics.csv", index=False)
    confusion.to_csv(out / "observed_05b_holdout_confusion.csv", index=False)
    classes.to_csv(out / "observed_05b_holdout_class_metrics.csv", index=False)
    analysis.paired_group_bootstrap(assigned, "previously_observed_05b_holdout").to_csv(
        out / "observed_05b_holdout_paired_group_bootstrap_ci.csv", index=False)
    analysis.leave_one_group_out(assigned, "previously_observed_05b_holdout").to_csv(
        out / "observed_05b_holdout_group_sensitivity.csv", index=False)
    analysis.distribution_summary(assigned, "previously_observed_05b_holdout").to_csv(
        out / "observed_05b_holdout_output_distributions.csv", index=False)
    missed = assigned.loc[(assigned.region == "near_failure") & (assigned.noul_value < analysis.N_LOW)]
    missed[["sample_id", "target", "region", "group_id", "ttf_seconds", "risk_score", "noul_value",
            "baseline_band", "candidate_band"]].to_csv(out / "observed_05b_holdout_gate_missed_near_failure.csv", index=False)
    oof = pd.read_csv(out / "development_cv_oof_predictions.csv")
    analysis.paired_group_bootstrap(oof, "development_group_cv_oof").to_csv(
        out / "development_cv_paired_group_bootstrap_ci.csv", index=False)
    all_rows = pd.concat([development, frame], ignore_index=True)
    analysis.plot_outputs(all_rows, out)
    usage = client.usage()
    prior = ctx["prior"]
    summary = {
        "historical_05b_read_only": {
            "requests": prior["api_calls_this_execution"],
            "cache_hits": prior["cache_hits_this_execution"],
            "parse_errors": prior["parse_errors_this_execution"],
            "input_tokens": prior["input_tokens_reported_this_execution"],
            "output_tokens": prior["output_tokens_reported_this_execution"],
            "reported_cost_usd": prior["reported_cost_usd_this_execution"],
        },
        "new_noul_followup": {**usage,
            "cache_hits_across_stages": int(development.cache_hit.sum()) + int(frame.cache_hit.sum()),
            "stage_request_attempts": {"pilot":45, "development_after_pilot":135, "observed_evaluation":90},
            "stage_cache_hits": {"pilot":0, "development":int(development.cache_hit.sum()),
                                 "observed_evaluation":int(frame.cache_hit.sum())}},
        "new_hard_caps": {"requests":300, "reported_cost_usd":0.05},
        "pilot_requests_included_in_new_total":45,
        "development_rows":180, "previously_observed_holdout_rows":90,
        "development_group_count":123, "previously_observed_holdout_group_count":57,
        "independent_holdout":False,
        "raw_feature_vectors_saved":False,
        "credentials_saved":False,
    }
    write_json(out / "usage_summary.json", summary)
    write_json(out / "run_status.json", {"stage":"observed_evaluation_complete",
               "evaluation_rows":len(frame), "development_rule_lock":str(lock_path.relative_to(ctx["root"])),
               "usage":usage, "independent_holdout":False})
    print("Locked-rule evaluation on previously observed 05b groups:",len(frame),"rows;",frame.group_id.nunique(),"groups.")
    print(metric.loc[metric.target == "macro"].to_string(index=False))
    print("Noul gate missed near-failure rows:",len(missed),"of",sum(frame.region == "near_failure"))
    print("New follow-up usage:",usage)
    print("Independent holdout:",False)
    return {"evaluation_rows":len(frame), "usage":usage, "independent_holdout":False}
