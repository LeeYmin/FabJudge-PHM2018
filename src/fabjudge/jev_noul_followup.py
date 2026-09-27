"""Exploratory paired Noul + exact v4 Score requests on the saved 05b cohort.

The cohort has been observed before. This module never labels its holdout independent.
Only local metadata and parsed responses are saved; feature vectors and credentials
are kept in memory. The earlier 05b artifacts are read-only inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "typesafe/jev-1.13"
ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
VERSION = "exact-v4-score+noul-current-condition-v1"
MODEL_PATTERN = re.compile(r"typesafe/jev-1\.13(?:-\d{8})?")
MAX_REQUESTS = 300
COST_CAP = 0.05
INITIAL_REQUEST_RESERVE = 0.00015
REGIONS = ("normal", "boundary", "near_failure")
NOUl_QUESTION = {
    "type": "noul",
    "instructions": "Using only the numeric measurements in state.features, do the measurements as a whole provide sufficient evidence of a current abnormal equipment condition?",
    "criteria": {
        "true": "The combined measurements support a present abnormal condition.",
        "false": "The combined measurements do not provide sufficient evidence of a present abnormal condition.",
    },
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_key(root: Path) -> str:
    """Load only the two approved variable names. Never return them in artifacts."""
    for name in ("JEV_API_KEY", "OPENROUTER_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    env_path = root / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if not item or item.startswith("#") or "=" not in item:
                continue
            name, value = item.split("=", 1)
            name = name.strip()
            if name in ("JEV_API_KEY", "OPENROUTER_API_KEY") and value.strip():
                os.environ.setdefault(name, value.strip().strip('"').strip("'"))
    return os.environ.get("JEV_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""


def load_context(root: Path):
    art = root / "artifacts"
    out = art / "jev_prompt_optimization" / "noul_followup" / "reused_groups_exploratory"
    out.mkdir(parents=True, exist_ok=True)
    base = read_json(art / "jev_tut" / "config.json")
    prior = read_json(art / "jev_prompt_optimization" / "config.json")
    scaler = read_json(art / "jev_tut" / "train_scaler_stats.json")
    v4 = json.loads((art / "jev_tut" / "prompt_v4.txt").read_text(encoding="utf-8").split("question=\n", 1)[1])
    assert v4 == read_json(art / "jev_prompt_optimization" / "decision_questions.json")["v4"]
    assert v4["type"] == "score" and len(v4["criteria"]) == 3
    assert prior["api_calls_this_execution"] == prior["max_api_calls"] == 552
    assert scaler["source"] == "03 train groups only"
    features = list(base["feature_columns"])
    if len(features) != 20 or len(set(features)) != 20:
        raise ValueError("The saved 20-feature specification changed.")
    sample = pd.read_csv(art / "jev_prompt_optimization" / "sample_audit.csv")
    if len(sample) != 270 or sample.sample_id.duplicated().any():
        raise ValueError("The saved 05b sample grid changed.")
    expected = {"development": 20, "holdout": 10}
    for subset, count in expected.items():
        cells = sample.loc[sample.subset == subset].groupby(["target", "region"]).size()
        if len(cells) != 9 or not (cells == count).all():
            raise ValueError(f"Unexpected saved {subset} cell counts.")
    dev = set(sample.loc[sample.subset == "development", "group_id"])
    observed = set(sample.loc[sample.subset == "holdout", "group_id"])
    if dev & observed:
        raise ValueError("Development and observed-holdout groups overlap.")
    physical = ["source_file", "sequence_index", "window_index"]
    if sample.duplicated(physical).any():
        raise ValueError("Saved 05b contains duplicate physical windows.")
    if not (sample.source_file.astype(str) + "::" + sample.sequence_index.astype(str) == sample.group_id).all():
        raise ValueError("Saved group IDs do not match source_file::sequence_index.")
    if set(sample.region) != set(REGIONS):
        raise ValueError("Unexpected TTF bands.")
    question = {"score": v4, "noul": NOUl_QUESTION}
    if (out / "questions.json").is_file():
        if read_json(out / "questions.json") != question:
            raise ValueError("Question version changed after requests began.")
    else:
        write_json(out / "questions.json", question)
    sample.to_csv(out / "sample_plan.csv", index=False)
    write_json(out / "sampling_audit.json", {
        "source": "previously observed 05b validation windows; exploratory reuse",
        "analysis_unit": "100 observations, stride 50, overlapping windows",
        "group_unit": "source_file::sequence_index",
        "development_rows": 180, "observed_holdout_rows": 90,
        "development_groups": len(dev), "observed_holdout_groups": len(observed),
        "group_overlap_between_subsets": 0, "duplicate_physical_windows": 0,
        "independent_holdout_claim": False,
        "reason": "All validation leak/near_failure groups available to 05b were previously sampled; this fallback reuses saved groups.",
    })
    return {"root": root, "out": out, "sample": sample, "features": features,
            "scaler": scaler, "questions": question, "prior": prior}


def reconstruct_features(ctx):
    """Read processed windows only for selected physical keys; never persist values."""
    root, sample, features = ctx["root"], ctx["sample"], ctx["features"]
    rows = {}
    for source, block in sample.groupby("source_file", sort=True):
        path = root / "data" / "interim" / "phm2018_jev" / "features" / "train" / source.replace(".csv", "_jev_windows.csv")
        if not path.is_file():
            raise FileNotFoundError(path)
        wanted = {(int(r.sequence_index), int(r.window_index)) for r in block.itertuples(index=False)}
        wanted_index = pd.MultiIndex.from_tuples(sorted(wanted))
        usecols = ["source_file", "sequence_index", "window_index", *features]
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=50_000, low_memory=False):
            seq = pd.to_numeric(chunk.sequence_index, errors="coerce").fillna(-1).astype(np.int64)
            win = pd.to_numeric(chunk.window_index, errors="coerce").fillna(-1).astype(np.int64)
            index = pd.MultiIndex.from_arrays([seq, win])
            chosen = chunk.loc[index.isin(wanted_index)]
            for item in chosen.itertuples(index=False):
                key = (str(item.source_file), int(item.sequence_index), int(item.window_index))
                if key in rows:
                    raise ValueError(f"Physical window appeared twice: {key}")
                values = {feature: float(getattr(item, feature)) for feature in features}
                if not all(math.isfinite(x) for x in values.values()):
                    raise ValueError(f"Non-finite selected feature vector: {key}")
                rows[key] = values
        print(f"Feature scan: {source}; selected physical windows {len(wanted)}", flush=True)
    expected = {(str(r.source_file), int(r.sequence_index), int(r.window_index))
                for r in sample.itertuples(index=False)}
    if set(rows) != expected:
        raise ValueError(f"Processed-window reconstruction missing {len(expected - set(rows))} selected rows.")
    return rows


def make_payload(ctx, row, raw_rows):
    values = raw_rows[(str(row.source_file), int(row.sequence_index), int(row.window_index))]
    scaled = {}
    for feature in ctx["features"]:
        spec = ctx["scaler"]["targets"][str(row.target)][feature]
        value = (values[feature] - float(spec["median_approx"])) / float(spec["safe_iqr"])
        if not math.isfinite(value):
            raise ValueError("Non-finite Robust value.")
        scaled[feature] = float(value)
    payload = {"model": MODEL, "state": {"features": scaled}, "questions": ctx["questions"]}
    if set(payload) != {"model", "state", "questions"} or set(payload["state"]) != {"features"}:
        raise AssertionError("Unexpected request fields.")
    if set(scaled) != set(ctx["features"]) or set(payload["questions"]) != {"score", "noul"}:
        raise AssertionError("The request must contain 20 numeric features and exactly two questions.")
    return payload


def parse_response(raw):
    if not isinstance(raw, dict):
        raise ValueError("Response is not an object.")
    model, provider = raw.get("model"), raw.get("provider")
    if not isinstance(model, str) or MODEL_PATTERN.fullmatch(model) is None or provider != "TypeSafe":
        raise ValueError("Returned model/provider differs from pinned Jev 1.13 / TypeSafe.")
    answers = raw.get("answers")
    if not isinstance(answers, dict) or set(answers) != {"score", "noul"}:
        raise ValueError("Both Score and Noul answers are required.")
    score_answer, noul_answer = answers["score"], answers["noul"]
    if score_answer.get("type") != "score" or noul_answer.get("type") != "noul":
        raise ValueError("Response question types differ from request types.")
    native_score, noul_value = float(score_answer["score"]), float(noul_answer["noul"])
    if not (math.isfinite(native_score) and 0 <= native_score <= 2):
        raise ValueError("Score is outside [0,2].")
    if not (math.isfinite(noul_value) and 0 <= noul_value <= 1):
        raise ValueError("Noul value is outside [0,1].")
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("Response usage object is missing.")
    input_tokens = int(usage["input_tokens"])
    output_tokens = int(usage["output_tokens"])
    cost = float(usage["cost"])
    if input_tokens < 0 or output_tokens < 0 or not math.isfinite(cost) or cost < 0:
        raise ValueError("Response token or cost usage is invalid.")
    return {"model": model, "provider": provider, "native_score": native_score,
            "risk_score": native_score / 2, "noul_value": noul_value,
            "usage_input_tokens": input_tokens, "usage_output_tokens": output_tokens,
            "usage_cost_usd": cost, "response_id": str(raw.get("id") or "")}


class Client:
    def __init__(self, ctx, key: str):
        self.ctx, self.key = ctx, key
        self.out = ctx["out"]
        self.cache = self.out / "cache"
        self.cache.mkdir(exist_ok=True)
        self.ledger = self.out / "api_attempts.jsonl"
        self.attempts = []
        if self.ledger.is_file():
            self.attempts = [json.loads(line) for line in self.ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(self.attempts) > MAX_REQUESTS:
            raise ValueError("Saved attempt count exceeds new-request cap.")
        self.calls_this_execution = 0
        self.cache_hits_this_execution = 0
        self.last_start = 0.0

    @property
    def spent(self):
        return float(sum(float(r.get("usage_cost_usd") or 0.0) for r in self.attempts))

    def _cache_path(self, sample_id, request_hash):
        identity = {"local_sample_id": sample_id, "model": MODEL,
                    "question_version": VERSION, "request_hash": request_hash}
        return self.cache / (hashlib.sha256(canonical(identity).encode("utf-8")).hexdigest() + ".json")

    def _cached(self, path, sample_id, request_hash):
        if not path.is_file():
            return None
        try:
            saved = read_json(path)
            valid = (saved.get("success") is True and saved.get("sample_id") == sample_id and
                     saved.get("request_hash") == request_hash and saved.get("question_version") == VERSION and
                     saved.get("provider") == "TypeSafe" and
                     MODEL_PATTERN.fullmatch(str(saved.get("model") or "")) is not None and
                     all(saved.get(k) is not None for k in ("risk_score", "noul_value", "usage_input_tokens", "usage_output_tokens", "usage_cost_usd")))
            if valid:
                self.cache_hits_this_execution += 1
                return {**saved, "cache_hit": True, "api_call": False}
        except (OSError, ValueError, TypeError):
            pass
        return None

    def invoke(self, row, raw_rows):
        payload = make_payload(self.ctx, row, raw_rows)
        request_hash = hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()
        sample_id = str(row.sample_id)
        cache_path = self._cache_path(sample_id, request_hash)
        cached = self._cached(cache_path, sample_id, request_hash)
        if cached is not None:
            return cached
        if len(self.attempts) >= MAX_REQUESTS:
            raise RuntimeError("New-request cap reached.")
        if self.spent + INITIAL_REQUEST_RESERVE > COST_CAP:
            raise RuntimeError("Reported-cost reserve would exceed the new hard cap.")
        if self.last_start:
            time.sleep(max(0.0, 0.25 - (time.perf_counter() - self.last_start)))
        body = canonical(payload).encode("utf-8")
        request = urllib.request.Request(ENDPOINT, data=body,
                    headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                    method="POST")
        self.last_start = time.perf_counter()
        self.calls_this_execution += 1
        status_code, raw, error = None, None, None
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                status_code = int(response.status)
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            status_code = int(exc.code)
            error = f"HTTP {status_code}"
            try:
                raw = json.loads(exc.read().decode("utf-8", errors="replace"))
            except ValueError:
                pass
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:160]}"
        parsed = None
        if status_code == 200:
            try:
                parsed = parse_response(raw)
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                error = f"{type(exc).__name__}: {str(exc)[:160]}"
        # Even a failed response may report billable usage. Record it without caching success.
        reported_usage = raw.get("usage") if isinstance(raw, dict) else None
        reported_usage = reported_usage if isinstance(reported_usage, dict) else {}
        try:
            reported_cost = float(reported_usage.get("cost"))
            if not math.isfinite(reported_cost) or reported_cost < 0:
                reported_cost = None
        except (TypeError, ValueError):
            reported_cost = None
        record = {"sample_id": sample_id, "question_version": VERSION, "requested_model": MODEL,
                  "request_hash": request_hash, "request_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                  "http_status": status_code, "api_call": True, "cache_hit": False,
                  "success": parsed is not None, "parse_success": parsed is not None,
                  "model": raw.get("model") if isinstance(raw, dict) else None,
                  "provider": raw.get("provider") if isinstance(raw, dict) else None,
                  "usage_cost_usd": reported_cost, "error": error or (None if parsed else "Invalid response")}
        if parsed:
            record.update(parsed)
        with self.ledger.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.attempts.append(record)
        if parsed:
            write_json(cache_path, record)
        return record

    def usage(self):
        rows = self.attempts
        return {"requests_cumulative": len(rows), "requests_this_execution": self.calls_this_execution,
                "cache_hits_this_execution": self.cache_hits_this_execution,
                "successful_responses_cumulative": sum(r.get("success") is True for r in rows),
                "parse_errors_cumulative": sum(r.get("parse_success") is not True for r in rows),
                "input_tokens_cumulative": sum(int(r.get("usage_input_tokens") or 0) for r in rows),
                "output_tokens_cumulative": sum(int(r.get("usage_output_tokens") or 0) for r in rows),
                "reported_cost_usd_cumulative": self.spent,
                "returned_models": pd.Series([r.get("model") for r in rows]).value_counts(dropna=False).to_dict() if rows else {},
                "providers": pd.Series([r.get("provider") for r in rows]).value_counts(dropna=False).to_dict() if rows else {}}


def run_pilot(root: Path):
    ctx = load_context(Path(root).resolve())
    out, sample = ctx["out"], ctx["sample"]
    print("Exploratory group reuse: development 180 rows / 123 groups; observed 05b holdout 90 rows / 57 groups.")
    print("The prior holdout is observed. This run cannot claim an independent evaluation.")
    print(sample.groupby(["subset", "target", "region"]).size().rename("rows").to_string())
    raw_rows = reconstruct_features(ctx)
    preview = make_payload(ctx, sample.sort_values("sample_id").iloc[0], raw_rows)
    write_json(out / "request_body_audit.json", {
        "model": MODEL, "endpoint": ENDPOINT, "top_level_fields": list(preview),
        "state_fields": list(preview["state"]), "question_types": {k:v["type"] for k,v in preview["questions"].items()},
        "feature_count": len(preview["state"]["features"]),
        "all_feature_values_numeric_finite": all(isinstance(v, float) and math.isfinite(v) for v in preview["state"]["features"].values()),
        "local_identifiers_targets_truth_ttf_region_filenames_group_ids_in_body": False,
        "sample_request_sha256": hashlib.sha256(canonical(preview).encode("utf-8")).hexdigest(),
        "feature_values_saved": False, "credentials_saved": False,
    })
    key = read_key(ctx["root"])
    if not key:
        write_json(out / "pilot_summary.json", {"status": "blocked_no_credential", "api_requests": 0})
        raise RuntimeError("JEV_API_KEY or OPENROUTER_API_KEY is not configured.")
    client = Client(ctx, key)
    pilot = (sample.loc[sample.subset == "development"].sort_values("sample_id")
             .groupby(["target", "region"], sort=True).head(5)
             .sort_values(["target", "region", "sample_id"]).reset_index(drop=True))
    if len(pilot) != 45 or pilot.sample_id.nunique() != 45:
        raise ValueError("Pilot must contain five distinct development samples in each of nine cells.")
    records = []
    for row in pilot.itertuples(index=False):
        rec = client.invoke(row, raw_rows)
        records.append({**rec, "target": row.target, "region": row.region, "group_id": row.group_id, "subset": row.subset})
        if not rec.get("success"):
            break
    pd.DataFrame(records).to_csv(out / "pilot_results.csv", index=False)
    max_seen_cost = max([float(r.get("usage_cost_usd") or 0.0) for r in client.attempts] + [0.0])
    reserve = max(0.0001, 1.5 * max_seen_cost)
    remaining_planned = 270 - len(records)
    projected_cost = client.spent + reserve * remaining_planned
    projected_calls = len(client.attempts) + remaining_planned
    passed = (len(records) == 45 and all(r.get("success") for r in records)
              and all(MODEL_PATTERN.fullmatch(str(r.get("model") or "")) is not None and r.get("provider") == "TypeSafe" for r in records)
              and all(all(r.get(k) is not None for k in ("usage_input_tokens", "usage_output_tokens", "usage_cost_usd")) for r in records)
              and projected_cost <= COST_CAP and projected_calls <= MAX_REQUESTS)
    summary = {"status": "passed" if passed else "failed", "pilot_samples": len(records),
               "pilot_expected": 45, "question_count_per_request": 2,
               "all_pilot_responses_successful": all(r.get("success") for r in records) if records else False,
               "model_counts": pd.Series([r.get("model") for r in records]).value_counts(dropna=False).to_dict() if records else {},
               "provider_counts": pd.Series([r.get("provider") for r in records]).value_counts(dropna=False).to_dict() if records else {},
               "reported_cost_usd_so_far": client.spent, "reserve_per_remaining_request_usd": reserve,
               "projected_total_reported_cost_usd": projected_cost,
               "projected_total_actual_request_attempts": projected_calls,
               "hard_call_cap": MAX_REQUESTS, "hard_cost_cap_usd": COST_CAP,
               "independent_holdout": False, "usage": client.usage()}
    write_json(out / "pilot_summary.json", summary)
    write_json(out / "run_status.json", {"stage": "pilot", "pilot_passed": passed,
                 "reused_groups_exploratory": True, "independent_holdout": False, "usage": client.usage()})
    print("Pilot:", len(records), "/45 paired requests; pass:", passed)
    print("Pilot returned models:", summary["model_counts"], "providers:", summary["provider_counts"])
    print("Pilot usage:", client.usage())
    print("Pilot projected total cost USD:", f"{projected_cost:.6f}", "calls:", projected_calls)
    return summary


def _network_budget_allows_completion(client: Client):
    """Conservative pilot-updated reserve for every uncached planned request."""
    cached_successes = len(list(client.cache.glob("*.json")))
    remaining = max(0, 270 - cached_successes)
    max_observed = max([float(r.get("usage_cost_usd") or 0.0) for r in client.attempts] + [0.0])
    reserve = max(0.0001, 1.5 * max_observed)
    projected_cost = client.spent + reserve * remaining
    projected_calls = len(client.attempts) + remaining
    return (projected_cost <= COST_CAP and projected_calls <= MAX_REQUESTS,
            {"cached_successes":cached_successes, "remaining_uncached":remaining,
             "reserve_usd_per_call":reserve, "projected_total_cost_usd":projected_cost,
             "projected_total_calls":projected_calls})


def _attach_local(record, row):
    local_fields = ("target", "target_label", "region", "subset", "source_file", "group_id",
                    "sequence_index", "window_index", "ttf_seconds")
    return {**record, **{field:getattr(row, field) for field in local_fields}}


def run_development(root: Path):
    from fabjudge import jev_noul_analysis as analysis
    ctx = load_context(Path(root).resolve())
    out, sample = ctx["out"], ctx["sample"]
    pilot = read_json(out / "pilot_summary.json")
    if pilot.get("status") != "passed":
        raise RuntimeError("The 45-request pilot did not pass.")
    plan = {
        "cohort":"previously observed 05b groups; exploratory only",
        "model":MODEL, "endpoint":ENDPOINT, "question_version":VERSION,
        "feature_count":20, "preprocessing":"05a train-only Robust, 1x",
        "response_model_pattern":MODEL_PATTERN.pattern, "provider":"TypeSafe",
        "baseline":"fresh same-request v4 Score / 2; three TTF surrogate classes",
        "candidate":"Noul <0.4 => low; 0.4<=Noul<=0.6 => review; Noul >0.6 => classify fresh v4 Score",
        "score_ties":"score <= t1 => normal; t1 < score <= t2 => boundary; score > t2 => near_failure",
        "missing":"either missing output => review; invalid responses stop the stage",
        "score_cut_search":"all ordered 0/1 endpoint and unique-score midpoint pairs, t1<t2",
        "score_cut_objective":"mean per-target balanced accuracy; tie: macro-F1, then lowest (t1,t2)",
        "group_cv":"3-fold StratifiedGroupKFold seed 42, all nine cells in train and test; fold training groups select both threshold pairs",
        "binary_noul_audit":"normal versus near_failure only; Noul >=0.5 is yes; TTF boundary excluded from this binary contrast",
        "evaluation":"rules locked after development; 05b holdout has already been observed and is not an independent holdout",
        "bootstrap":"paired group resampling with whole groups, both predictions; point estimate from raw rows",
        "hard_limits":{"requests":MAX_REQUESTS, "reported_cost_usd":COST_CAP},
        "target_label_limit":"TTF bands are surrogate labels, not current fault truth; balanced precision and PR-AUC do not estimate operating prevalence",
    }
    plan_path = out / "analysis_plan_locked_before_development.json"
    if plan_path.is_file():
        if read_json(plan_path) != plan:
            raise ValueError("Locked analysis plan changed after pilot.")
    else:
        write_json(plan_path, plan)
    raw_rows = reconstruct_features(ctx)
    key = read_key(ctx["root"])
    if not key:
        raise RuntimeError("JEV_API_KEY or OPENROUTER_API_KEY is not configured.")
    client = Client(ctx, key)
    development = (sample.loc[sample.subset == "development"]
                   .sort_values(["target", "region", "sample_id"]).reset_index(drop=True))
    records = []
    for row in development.itertuples(index=False):
        allowed, projection = _network_budget_allows_completion(client)
        if not allowed:
            write_json(out / "run_status.json", {"stage":"development_budget_stop",
                       "projection":projection, "usage":client.usage(), "independent_holdout":False})
            raise RuntimeError("Remaining planned requests no longer fit the new hard limits.")
        rec = client.invoke(row, raw_rows)
        records.append(_attach_local(rec, row))
        if not rec.get("success"):
            break
    frame = pd.DataFrame(records)
    frame.to_csv(out / "development_responses.csv", index=False)
    if len(frame) != 180 or not frame.success.all():
        write_json(out / "run_status.json", {"stage":"development_incomplete",
                   "development_rows":len(frame), "usage":client.usage(), "independent_holdout":False})
        raise RuntimeError("Development responses are incomplete or invalid; observed holdout remains untouched by this run.")
    if frame.sample_id.duplicated().any() or frame.group_id.nunique() != 123:
        raise ValueError("Saved development sample pairing or group count changed.")
    continuous = analysis.continuous_development(frame)
    continuous.to_csv(out / "development_continuous_score_noul.csv", index=False)
    thresholds = {method:analysis.select_thresholds(frame, method) for method in ("baseline", "candidate")}
    apparent = analysis.assign_predictions(frame, thresholds)
    oof, fold_rules = analysis.group_cv(frame)
    fold_rules.to_csv(out / "development_cv_fold_thresholds.csv", index=False)
    oof.to_csv(out / "development_cv_oof_predictions.csv", index=False)
    apparent.to_csv(out / "development_predictions.csv", index=False)
    metric, confusion, classes = analysis.three_class_tables(apparent, "development_apparent")
    cv_metric, cv_confusion, cv_classes = analysis.three_class_tables(oof, "development_group_cv_oof")
    pd.concat([metric, cv_metric], ignore_index=True).to_csv(out / "development_metrics.csv", index=False)
    pd.concat([confusion, cv_confusion], ignore_index=True).to_csv(out / "development_confusion.csv", index=False)
    pd.concat([classes, cv_classes], ignore_index=True).to_csv(out / "development_class_metrics.csv", index=False)
    analysis.paired_group_bootstrap(apparent, "development_apparent").to_csv(out / "development_paired_group_bootstrap_ci.csv", index=False)
    analysis.leave_one_group_out(apparent, "development_apparent").to_csv(out / "development_group_sensitivity.csv", index=False)
    analysis.distribution_summary(apparent, "development_apparent").to_csv(out / "development_output_distributions.csv", index=False)
    missed = apparent.loc[(apparent.region == "near_failure") & (apparent.noul_value < analysis.N_LOW)]
    missed[["sample_id", "target", "region", "group_id", "ttf_seconds", "risk_score", "noul_value",
            "baseline_band", "candidate_band"]].to_csv(out / "development_gate_missed_near_failure.csv", index=False)
    lock = {"locked_before_new_evaluation_requests":True, "cohort":"development_apparent",
            "thresholds":thresholds, "noul_review_band_inclusive":[analysis.N_LOW, analysis.N_HIGH],
            "group_cv_folds":3, "independent_holdout_claim":False,
            "old_05b_holdout_already_observed":True,
            "lock_time_utc":pd.Timestamp.now(tz="UTC").isoformat()}
    lock_path = out / "development_rule_lock.json"
    if lock_path.is_file():
        previous = read_json(lock_path)
        if previous.get("thresholds") != thresholds:
            raise ValueError("Existing development rule lock differs; evaluation cannot proceed.")
    else:
        write_json(lock_path, lock)
    write_json(out / "run_status.json", {"stage":"development_complete_rules_locked",
               "development_rows":len(frame), "group_cv_rows":len(oof),
               "usage":client.usage(), "independent_holdout":False})
    print("Development responses complete:",len(frame),"; groups:",frame.group_id.nunique())
    print("Development continuous comparison:")
    print(continuous.loc[continuous.target == "macro"].to_string(index=False))
    print("Locked thresholds:",thresholds)
    print("Development and grouped-CV macro metrics:")
    print(pd.concat([metric, cv_metric]).loc[lambda d:d.target == "macro"].to_string(index=False))
    print("Development usage:",client.usage())
    return {"thresholds":thresholds, "usage":client.usage(), "development_rows":len(frame)}
