from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / ".venv" / "Lib" / "site-packages"
if SITE.is_dir():
    sys.path.insert(0, str(SITE))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from fabjudge.jev_v4_validation import validate_candidate

MODEL = "typesafe/jev-1.13"
URL = "https://openrouter.ai/api/alpha/decisions"
TARGETS = ("ttf_flowcool_leak_seconds", "ttf_flowcool_pressure_high_seconds", "ttf_flowcool_pressure_low_seconds")
BANDS = ("normal", "boundary", "near_failure")
OUT = ROOT / "artifacts" / "jev_v4_edge_trials"
DATA = ROOT / "data" / "interim" / "phm2018_jev" / "features" / "train"

CANDIDATES = [
("V4E01_BIDIRECTIONAL_DEVIATION","Symmetric positive and negative departures from feature centers.","05b v4 leak normal Score 0.675 exceeded near-failure 0.320; macro ROC-AUC 0.6017.","Reduce missed low-side near-failure; may raise normal-to-high.","Interpret each Robust-scaled feature relative to its own training center: positive and negative departures of similar magnitude can both indicate abnormality; do not assume higher is always worse."),
("V4E02_SENSOR_EVIDENCE_DEDUP","Count summaries from one physical sensor as one evidence source.","05b v4 pressure-high normal 0.760 exceeded near-failure 0.375; possible summary dominance.","Reduce normal-to-high from duplicate summaries; may suppress complementary fault evidence.","Features sharing a physical-variable prefix are alternate summaries of one sensor; count them as one evidence source, not several independent votes."),
("V4E03_PEAK_TRANSIENT_CONTROL","Prioritize sustained mean/RMS summaries over an isolated peak.","05b grouped-CV leak near-failure recall was 3/20; ordinary-state high scores also occurred.","Reduce isolated-peak alarms; may miss peak-only early faults.","Give __mean and __rms more evidential weight than a lone __peak_abs excursion; treat an isolated peak as weak evidence unless sustained measurements also support it."),
("V4E04_LEVEL_VS_DISPERSION","Treat level shift and within-window variability as distinct signals.","05b grouped-CV three-band recall was weak for normal and near-failure.","Recover variability-led failures; may raise normal-to-high.","Evaluate __mean as operating level and __rms as variability; a typical mean does not cancel abnormal variability, and large RMS alone does not establish abnormality."),
("V4E05_SHAPE_FACTOR_SEMANTICS","Interpret shape_factor as distribution shape, not equipment level.","05b v4 leak ordering was weak with shape_factor among the fixed features.","Recover shape-led failures; may over-read benign variation.","Read __shape_factor only as a descriptor of within-window distribution shape, not as the equipment measurement level itself."),
("V4E06_SKEWNESS_TAIL_ASYMMETRY","Use skewness as a cue about asymmetric within-window tails.","05b v4 had cross-class ties and a leak normal-above-near reversal.","Improve ordering where asymmetric tails precede mean shifts; may flag naturally skewed normal windows.","Interpret __skewness as the direction and strength of within-window tail asymmetry; asymmetry can support abnormality even when the mean is modest."),
 ("V4E07_COOLING_PRESSURE_FLOW_COUPLING","Interpret cooling pressure and flow rate as a coupled equipment relationship.","05b v4 leak normal Score 0.675 exceeded near-failure 0.320; cooling pressure and flow-rate evidence may decouple as a leak develops.","Reduce missed leak near-failure when pressure-flow consistency shifts; may flag benign operating changes that alter their relationship.","Evaluate FLOWCOOLPRESSURE and FLOWCOOLFLOWRATE jointly as a coupled cooling system; a mismatch between their departures can support abnormality even when neither is extreme alone."),
("V4E08_ORDINAL_SCORE_ANCHORS","Bind Score levels explicitly to numeric anchors 0, 1, and 2.","05b macro ROC-AUC was 0.6017 and macro separation was 0.0337.","Improve ordering with explicit anchors; may compress intermediate cases.","Use 0 for normal, 1 for uncertain or mixed evidence, and 2 for strongly abnormal evidence; intermediate Score values express ordered severity between these anchors."),
("V4E09_SENSOR_DIRECTION_SEMANTICS","Use named physical measurement semantics to identify high/low direction.","05b v4 leak normal 0.675 exceeded near-failure 0.320.","Reduce errors from hidden low-side abnormality; may add false direction assumptions.","Use each named physical measurement to judge whether its high or low direction is abnormal; do not treat either tail as universally more severe."),
("V4E10_CONTINUOUS_SCORE_RESOLUTION","Preserve fine-grained Score differences within each ordinal level.","05b v4 had cross-class ties in all targets and macro ROC-AUC 0.6017.","Reduce tied pairs; may add unstable small score shifts.","Use the full continuous Score range to preserve differences in evidence strength within a state level rather than assigning every case in a level the same typical Score.")
]


def _hash(value):
    return hashlib.sha256(value if isinstance(value, bytes) else str(value).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _v4(root):
    question = json.loads((root / "artifacts/jev_tut/prompt_v4.txt").read_text(encoding="utf-8").split("question=\n", 1)[1])
    if question != _read(root / "artifacts/jev_prompt_optimization/decision_questions.json")["v4"]:
        raise ValueError("Saved exact v4 question mismatch.")
    return question


def _spec():
    return {
        "spec_version": "1.2", "model": MODEL, "targets": list(TARGETS),
        "group_unit": "source_file::sequence_index",
        "class_mapping": {"normal": 0, "boundary": 1, "near_failure": 2},
        "ttf_bands_seconds": {"near_failure": "<=5000", "boundary": "5000<TTF<=50000", "normal": ">50000"},
        "score": {"analysis": "native Score / 2", "range": [0, 1], "not_failure_probability": True},
        "continuous_metric": {"per_target": "ROC-AUC; near_failure positive, normal negative", "boundary": "excluded", "macro": "unweighted target mean"},
        "threshold_cv": {
            "method": "StratifiedGroupKFold", "n_splits": 3, "shuffle": True, "seed": 42,
            "strata": "target|band", "fit": "per-target thresholds from fold training groups only",
            "candidates": "0 and 1 endpoints plus midpoints of adjacent unique training scores",
            "mapping": "score<=t1 normal; t1<score<=t2 boundary; score>t2 near_failure",
            "objective": "per-target three-class macro-F1 on training groups",
            "tie_break": "lexicographically smallest (t1,t2)", "evaluation": "OOF macro-F1 per target, then unweighted mean",
            "missing_class": "INCONCLUSIVE",
        },
        "dangerous_errors": {"normal_to_high": "true normal predicted near_failure; pooled and per-target rates/counts",
                             "near_failure_to_low": "true near_failure predicted normal; pooled and per-target rates/counts"},
        "group_sensitivity": "same OOF folds and training-only threshold search after removing each whole group",
        "provisional_keep_rule": {
            "roc_candidate_min": 0.62, "roc_delta_min": 0.02, "f1_candidate_min": 0.45, "f1_delta_min": 0.03,
            "safety_roc_tolerance": 0.01, "safety_f1_tolerance": 0.01,
            "normal_to_high_rate_tolerance": 0.05, "near_failure_to_low_count_must_not_increase": True,
            "logic": "(roc_branch OR f1_branch) AND safety", "scope": "development-only",
        },
        "numeric_comparison": {"inclusive_float_epsilon": 1e-12,
                                "reason": "Inclusive threshold comparisons use this tolerance only to neutralize binary floating-point boundary representation; metric values and user thresholds are unchanged."},
        "final_confirmation": {
            "primary_metric": "macro_roc_auc", "paired_group_bootstrap_replicates": 2000, "seed": 42, "ci": 0.95,
            "selection": "highest development macro ROC-AUC among provisional keeps; tie by OOF macro-F1, then earlier trial",
            "insufficient_unseen_groups": "INCONCLUSIVE, never VERIFIED",
        },
        "sampling": {"source": "05b development rows", "per_target_band": 5,
                     "selection": "SHA256 rank, one row per group in each cell", "seed": 42,
                     "selection_bias": "same windows reused across trials"},
        "request": {"fields": ["model", "state", "questions"], "state_fields": ["features"],
                    "questions": ["v4", "candidate"], "feature_count": 20,
                    "no_ids_targets_ttf_labels_groups": True, "save_raw_feature_vectors": False, "cache_failures": False},
        "caps": {"requests": 600, "reported_cost_usd": 0.05},
    }


def _make_split(sample):
    y = (sample.target.astype(str) + "|" + sample.region.astype(str)).to_numpy()
    groups = sample.group_id.astype(str).to_numpy()
    ids = sample.sample_id.astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    folds = []
    for i, (train_ix, valid_ix) in enumerate(splitter.split(sample, y, groups)):
        train, valid = sample.iloc[train_ix], sample.iloc[valid_ix]
        if set(train.group_id.astype(str)) & set(valid.group_id.astype(str)):
            raise ValueError("Group leakage.")
        for target in TARGETS:
            if set(train.loc[train.target == target, "region"]) != set(BANDS):
                raise ValueError("A fold training partition is missing a class.")
        folds.append({"fold": i, "train_sample_ids": sorted(ids[train_ix].tolist()),
                      "validation_sample_ids": sorted(ids[valid_ix].tolist())})
    return {"group_unit": "source_file::sequence_index", "rule": "StratifiedGroupKFold(3, shuffle=True, seed=42), strata=target|band", "folds": folds}


def prepare_experiment(root=ROOT):
    root = Path(root).resolve()
    out = root / "artifacts/jev_v4_edge_trials"
    for name in ("responses", "validations", "success_cache"):
        (out / name).mkdir(parents=True, exist_ok=True)
    question = _v4(root)
    config = _read(root / "artifacts/jev_tut/config.json")
    features = list(config["feature_columns"])
    if len(features) != 20 or config.get("model_id") != MODEL:
        raise ValueError("Existing model/feature audit failed.")
    pool = pd.read_csv(root / "artifacts/jev_prompt_optimization/sample_audit.csv")
    pool = pool.loc[pool.subset == "development"].copy()
    chosen = []
    for target in TARGETS:
        for band in BANDS:
            cell = pool.loc[(pool.target == target) & (pool.region == band)].copy()
            cell["_rank"] = cell.sample_id.astype(str).map(lambda x: _hash(f"42|{target}|{band}|{x}"))
            cell = cell.sort_values(["_rank", "sample_id"]).drop_duplicates("group_id").head(5)
            if len(cell) != 5:
                raise ValueError(f"Insufficient development groups for {target}/{band}.")
            chosen.extend(cell.drop(columns="_rank").to_dict("records"))
    sample = pd.DataFrame(chosen).sort_values(["target", "region", "sample_id"]).reset_index(drop=True)
    plan_path = out / "development_sample_plan.csv"
    sample[["sample_id", "target", "region", "subset", "source_file", "group_id", "sequence_index", "window_index", "ttf_seconds"]].to_csv(plan_path, index=False)
    split = _make_split(sample)
    split_path = out / "group_split.json"
    split_path.write_text(json.dumps(split, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    spec_bytes = (json.dumps(_spec(), sort_keys=True, indent=2) + "\n").encode()
    spec_path = out / "validation_spec.json"
    manifest_path = out / "frozen_manifest.json"
    if spec_path.exists() and spec_path.read_bytes() != spec_bytes and manifest_path.exists():
        raise RuntimeError("Refusing to overwrite a frozen spec.")
    if not spec_path.exists() or spec_path.read_bytes() != spec_bytes:
        spec_path.write_bytes(spec_bytes)
    spec_hash = _hash(spec_path.read_bytes())
    (out / "validation_spec.sha256").write_text(spec_hash + "  validation_spec.json\n", encoding="ascii")
    bank = {item[0]: {"type": "score", "instructions": question["instructions"] + " " + item[4], "criteria": question["criteria"]} for item in CANDIDATES}
    bank_path = out / "candidate_questions.json"
    _write(bank_path, bank)
    old = pd.read_csv(root / "artifacts/jev_prompt_optimization/all_results.csv",
                      usecols=["sample_id", "prompt_id", "risk_score", "subset", "target", "region", "group_id"])
    old = old.loc[(old.subset == "development") & old.sample_id.isin(sample.sample_id)]
    paired = old.pivot(index=["sample_id", "target", "region", "group_id"], columns="prompt_id", values="risk_score").reset_index()
    paired = paired.rename(columns={"region": "band", "v4": "v4_score", "v5": "candidate_score"})
    preflight = validate_candidate(paired.to_dict("records"), split, _read(spec_path))
    if preflight["decision"] == "INCONCLUSIVE":
        raise RuntimeError("Saved-response validator preflight inconclusive: " + str(preflight["reasons"]))
    _write(out / "validator_preflight_old_05b.json", {"use": "offline implementation preflight only; not a trial",
           "decision": preflight["decision"], "spec_sha256": spec_hash, "sample_count": preflight["sample_count"],
           "metrics": preflight["metrics"], "checks": preflight["checks"]})
    prior = [
        ("PRIOR_05A_BASELINE_SCALERS", "Generic v4 Score and RAW/Z-score/Robust comparison.", "05a Robust reached ROC-AUC 0.913 on a separate cohort; prompt stayed generic.", "Representation-only, not a new prompt mechanism.", "N/A", "05a"),
        ("PRIOR_SCALE_FACTORS", "Single-feature scaling at 0.5x/2x and 0.1x/10x.", "Exact v4 prompt and 1x control retained.", "Scale-only hypothesis excluded.", "N/A", "scale sensitivity"),
        ("PRIOR_05B_V5_PATTERN_UNCERTAINTY", "Current-condition wording, overall pattern, avoid one value, mixed evidence as Uncertain.", "v5 failed the locked development rule; v4 retained.", "No wording-only variant will be recounted.", "v5 reduced ROC-AUC/separation.", "05b"),
        ("PRIOR_THRESHOLD_DIAGNOSTICS", "Three-stage Score cutoffs and grouped CV.", "Prior OOF F1 modest; threshold intervals wide.", "Threshold tuning is not a new prompt mechanism.", "Threshold overfit.", "diagnostics"),
        ("PRIOR_NOUL_GATE", "Noul low/review/high gate around v4 Score.", "Observed 05b holdout Noul candidate reduced performance and missed near-failure.", "Noul gate excluded.", "Near-failure-to-low errors increased.", "Noul"),
    ]
    registry = out / "hypothesis_registry.csv"
    fields = ["trial", "mechanism_id", "changed_element", "evidence_v4_misclassification", "expected_improve_error", "expected_worsen_error", "duplicate_of", "status", "source"]
    if not registry.exists():
        rows = [{"trial": "", "mechanism_id": a, "changed_element": b, "evidence_v4_misclassification": c,
                 "expected_improve_error": d, "expected_worsen_error": e, "duplicate_of": "",
                 "status": "ALREADY_TESTED_NOT_RECOUNTED", "source": f} for a, b, c, d, e, f in prior]
        _csv(registry, rows, fields)
    feasibility = _read(root / "artifacts/jev_prompt_optimization/noul_followup/feasibility_status.json")
    manifest = {"spec_sha256": spec_hash, "validator_sha256": _hash((root / "src/fabjudge/jev_v4_validation.py").read_bytes()),
                "runner_sha256": _hash(Path(__file__).read_bytes()), "split_sha256": _hash(split_path.read_bytes()),
                "sample_sha256": _hash(plan_path.read_bytes()), "questions_sha256": _hash(bank_path.read_bytes()),
                "development_samples": len(sample), "development_groups": int(sample.group_id.nunique()),
                "final_group_feasible": bool(feasibility.get("new_balanced_nine_cell_design", {}).get("feasible", False)),
                "created_utc": datetime.now(timezone.utc).isoformat()}
    if manifest_path.exists():
        saved = _read(manifest_path)
        if any(saved.get(k) != manifest[k] for k in ("spec_sha256", "validator_sha256", "runner_sha256", "split_sha256", "sample_sha256", "questions_sha256")):
            usage_path = out / "usage_summary.json"
            used = _read(usage_path).get("api_requests", 0) if usage_path.exists() else 0
            if used != 0 or (out / "api_attempts.jsonl").exists():
                raise RuntimeError("Frozen input changed after API use.")
            _write(manifest_path, manifest)
        else:
            manifest = saved
    else:
        _write(manifest_path, manifest)
    if not (out / "usage_summary.json").exists():
        _write(out / "usage_summary.json", {"api_requests": 0, "success_responses": 0, "failed_attempts": 0, "cache_hits": 0,
               "input_tokens": 0, "output_tokens": 0, "reported_cost_usd": 0.0, "max_unit_cost": 0.0,
               "caps": {"requests": 600, "reported_cost_usd": 0.05}, "prior_usage_included": False})
    reason = feasibility.get("reason", "")
    if not (out / "trial_log.md").exists() or _read(out / "usage_summary.json").get("api_requests", 0) == 0:
        text = ("# JEV v4 Score edge trials\n\n"
                f"- Frozen validation spec SHA256: {spec_hash}\n- Frozen validator SHA256: {manifest['validator_sha256']}\n"
                "- Pre-API version history: v1.0 preflight input alias mismatch (`region`/`band`) detected; versioned to v1.1 and all saved development v4/v5 pairs re-evaluated offline.\n"
                "- Pre-API registry review removed the scale-context proposal as a duplicate of the prior scaling study; trial 7 is a cooling pressure-flow coupling hypothesis.\n"
                "- 05b and prior artifacts are read-only inputs.\n"
                f"- Development: {len(sample)} paired windows / {sample.group_id.nunique()} groups; five per target x band cell.\n"
                "- Fixed 3-fold StratifiedGroupKFold, seed 42; fold training groups alone fit thresholds.\n"
                "- New-call cap: 600 requests and USD 0.05, separate from historical usage.\n"
                f"- Final independent groups: INCONCLUSIVE: {reason}\n"
                "- Reusing development windows across trials is selection-biased.\n\n"
                "| trial | new mechanism_id | changed element | sample/groups | candidate/v4 ROC-AUC | candidate/v4 OOF macro-F1 | dangerous error differences | roc_branch | f1_branch | safety | validator verdict | cumulative calls/cost |\n"
                "|---:|---|---|---:|---:|---:|---|---|---|---|---|---|\n")
        (out / "trial_log.md").write_text(text, encoding="utf-8")
    notebook = root / "notebooks/05c_jev_v4_edge_trials.ipynb"
    if not notebook.exists():
        cells = [
            {"cell_type": "markdown", "metadata": {}, "source": [
                f"# 05c JEV v4 Score edge trials\n\nFrozen spec SHA256: {spec_hash}.\n\n",
                "Isolated from 05b. Fixed 20 features, train-only Robust 1x, same-request exact-v4/candidate Score pairs. The same development windows are reused, so results are exploratory and selection-biased.\n\n",
                "Preflight found no unused validation groups in the FlowCool-leak/near-failure cell. Final confirmation is INCONCLUSIVE without eligible unseen groups.\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "from pathlib import Path\n", "import sys\n", "ROOT=Path.cwd().resolve(); sys.path.insert(0,str(ROOT/'src'))\n",
                "from fabjudge.jev_v4_edge_trials import prepare_experiment, run_trial, finalize_experiment\n", "prepare_experiment(ROOT)\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["Run one registered trial at a time in order. Each run saves paired responses, validation, usage, and the Markdown report row.\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "# Example: run_trial(1, ROOT)\n", "import pandas as pd\n",
                "p=ROOT/'artifacts/jev_v4_edge_trials/trial_summary.csv'\n", "pd.read_csv(p) if p.exists() else pd.DataFrame()\n"]},
        ]
        notebook.parent.mkdir(parents=True, exist_ok=True)
        notebook.write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}, ensure_ascii=True, indent=1) + "\n", encoding="utf-8")
    return {"spec_sha256": spec_hash, "validator_sha256": manifest["validator_sha256"], "samples": len(sample),
            "groups": int(sample.group_id.nunique()), "old_response_preflight": preflight["decision"],
            "final_groups_feasible": manifest["final_group_feasible"]}


def _credential(root):
    for name in ("JEV_API_KEY", "OPENROUTER_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    env_file = root / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                name, value = line.split("=", 1)
                if name.strip() in ("JEV_API_KEY", "OPENROUTER_API_KEY") and value.strip().strip("\"'"):
                    return value.strip().strip("\"'")
    return None


def _frozen(root, out):
    manifest = _read(out / "frozen_manifest.json")
    spec_path = out / "validation_spec.json"
    if _hash(spec_path.read_bytes()) != manifest["spec_sha256"] or (out / "validation_spec.sha256").read_text().split()[0] != manifest["spec_sha256"]:
        raise RuntimeError("Validation spec hash mismatch.")
    checks = [("validator_sha256", root / "src/fabjudge/jev_v4_validation.py"), ("runner_sha256", Path(__file__)),
              ("split_sha256", out / "group_split.json"), ("sample_sha256", out / "development_sample_plan.csv"),
              ("questions_sha256", out / "candidate_questions.json")]
    for key, path in checks:
        if _hash(path.read_bytes()) != manifest[key]:
            raise RuntimeError("Frozen file changed: " + key)
    return manifest, _read(spec_path)


def _raw_features(plan, columns):
    output = {}
    for source, rows in plan.groupby("source_file"):
        path = DATA / str(source).replace(".csv", "_jev_windows.csv")
        wanted = {(int(r.sequence_index), int(r.window_index)) for r in rows.itertuples()}
        for chunk in pd.read_csv(path, usecols=["source_file", "sequence_index", "window_index", *columns], chunksize=25000):
            mask = [(int(a), int(b)) in wanted for a, b in zip(chunk.sequence_index, chunk.window_index)]
            for row in chunk.loc[mask].itertuples(index=False):
                key = (str(row.source_file), int(row.sequence_index), int(row.window_index))
                output[key] = {c: float(getattr(row, c)) for c in columns}
    expected = {(str(r.source_file), int(r.sequence_index), int(r.window_index)) for r in plan.itertuples()}
    if set(output) != expected:
        raise ValueError("Could not reconstruct selected processed windows.")
    return output


def _request(payload, key):
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    req = urllib.request.Request(URL, data=body, headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
    started = time.time()
    status, obj, error = None, None, None
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            status, obj = response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        status = exc.code
        try:
            obj = json.loads(exc.read().decode())
        except Exception:
            error = "HTTPError"
    except Exception as exc:
        error = type(exc).__name__
    model = obj.get("model") if isinstance(obj, dict) else None
    provider = obj.get("provider") if isinstance(obj, dict) else None
    usage = obj.get("usage", {}) if isinstance(obj, dict) else {}
    answers = obj.get("answers", {}) if isinstance(obj, dict) else {}
    model_ok = bool(re.fullmatch(r"typesafe/jev-1\.13(?:-\d{8})?", str(model or "")))
    scores, parse_ok = {}, True
    for question in ("v4", "candidate"):
        answer = answers.get(question, {}) if isinstance(answers, dict) else {}
        try:
            value = float(answer.get("score"))
            valid = answer.get("type") == "score" and math.isfinite(value) and 0 <= value <= 2
        except Exception:
            value, valid = None, False
        scores[question] = value
        parse_ok = parse_ok and valid
    try:
        input_tokens, output_tokens, cost = int(usage["input_tokens"]), int(usage["output_tokens"]), float(usage["cost"])
        usage_ok = input_tokens >= 0 and output_tokens >= 0 and math.isfinite(cost) and cost >= 0
    except Exception:
        input_tokens = output_tokens = cost = None
        usage_ok = False
    success = status == 200 and model_ok and provider == "TypeSafe" and parse_ok and usage_ok
    if not success and not error:
        error = "model_mismatch" if not model_ok else "provider_mismatch" if provider != "TypeSafe" else "score_parse_error" if not parse_ok else "usage_invalid"
    return {"success": success, "status": status, "model": model, "provider": provider,
            "scores": scores if success else None, "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cost": cost, "latency": time.time() - started, "response_id": obj.get("id") if isinstance(obj, dict) else None,
            "error": error}


def _usage(out):
    return _read(out / "usage_summary.json")


def run_trial(number, root=ROOT):
    root = Path(root).resolve()
    out = root / "artifacts/jev_v4_edge_trials"
    manifest, validation_spec = _frozen(root, out)
    number = int(number)
    if not 1 <= number <= 10:
        raise ValueError("Trial number must be 1..10.")
    for earlier in range(1, number):
        path = out / "validations" / f"trial_{earlier:02d}.json"
        if not path.exists() or _read(path).get("decision") not in ("REJECT", "PROVISIONAL_KEEP"):
            raise ValueError(f"Trial {earlier} must finish first.")
    result_path = out / "validations" / f"trial_{number:02d}.json"
    if result_path.exists() and _read(result_path).get("decision") in ("REJECT", "PROVISIONAL_KEEP"):
        raise ValueError("Completed trial is frozen.")
    item = CANDIDATES[number - 1]
    registry_path = out / "hypothesis_registry.csv"
    registry = list(csv.DictReader(registry_path.open(encoding="utf-8", newline="")))
    if not any(x["mechanism_id"] == item[0] for x in registry):
        evidence = item[2]
        if number > 1:
            previous = _read(out / "validations" / f"trial_{number - 1:02d}.json")
            if previous.get("metrics"):
                m = previous["metrics"]["v4"]
                weak = min(m["per_target"], key=lambda t: m["per_target"][t]["roc_auc_normal_vs_near_failure"])
                err = m["dangerous_errors"]
                evidence = (f"Prior same-request v4 residual: weakest {weak} AUC={m['per_target'][weak]['roc_auc_normal_vs_near_failure']:.4f}; "
                            f"normal-to-high {err['normal_to_high']['count']}/{err['normal_to_high']['denominator']}, "
                            f"near-to-low {err['near_failure_to_low']['count']}/{err['near_failure_to_low']['denominator']}.")
        registry.append({"trial": number, "mechanism_id": item[0], "changed_element": item[1],
                         "evidence_v4_misclassification": evidence, "expected_improve_error": item[3].split(";")[0],
                         "expected_worsen_error": item[3].split(";")[-1], "duplicate_of": "",
                         "status": "REGISTERED_BEFORE_API", "source": "same-request v4 development residual"})
        _csv(registry_path, registry, list(registry[0].keys()))
    base = _v4(root)
    candidate = _read(out / "candidate_questions.json")[item[0]]
    plan = pd.read_csv(out / "development_sample_plan.csv")
    columns = _read(root / "artifacts/jev_tut/config.json")["feature_columns"]
    scaler = _read(root / "artifacts/jev_tut/train_scaler_stats.json")
    raw = _raw_features(plan, columns)
    credential = _credential(root)
    usage = _usage(out)
    paired, failure = [], None
    if not credential:
        failure = "No OpenRouter API credential configured."
    reserve = max(0.000075, float(usage.get("max_unit_cost", 0)) * 1.5)
    planned = (10 - number + 1) * len(plan) + len(plan)
    if usage["api_requests"] + planned > 600 or usage["reported_cost_usd"] + reserve * planned > 0.05:
        failure = "Projected budget exceeds a hard cap."
    if not failure:
        for row in plan.itertuples(index=False):
            target = str(row.target)
            vector = raw[(str(row.source_file), int(row.sequence_index), int(row.window_index))]
            features = {}
            for column in columns:
                state = scaler["targets"][target][column]
                value = (vector[column] - float(state["median_approx"])) / float(state["safe_iqr"])
                if not math.isfinite(value):
                    failure = "Non-finite Robust feature."
                    break
                features[column] = value
            if failure:
                break
            payload = {"model": MODEL, "state": {"features": features}, "questions": {"v4": base, "candidate": candidate}}
            if set(payload) != {"model", "state", "questions"} or set(payload["state"]) != {"features"}:
                raise ValueError("Request contains forbidden metadata.")
            request_hash = _hash(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
            cache_path = out / "success_cache" / (request_hash + ".json")
            cached = _read(cache_path) if cache_path.exists() else None
            if cached and cached.get("success") and cached.get("provider") == "TypeSafe":
                response = cached
                usage["cache_hits"] += 1
            else:
                response = _request(payload, credential)
                usage["api_requests"] += 1
                if response["cost"] is not None:
                    usage["reported_cost_usd"] += response["cost"]
                    usage["max_unit_cost"] = max(usage.get("max_unit_cost", 0), response["cost"])
                if response["input_tokens"] is not None:
                    usage["input_tokens"] += response["input_tokens"]
                if response["output_tokens"] is not None:
                    usage["output_tokens"] += response["output_tokens"]
                if response["success"]:
                    usage["success_responses"] += 1
                    _write(cache_path, {"success": True, "request_hash": request_hash, "model": response["model"],
                            "provider": response["provider"], "scores": response["scores"], "input_tokens": response["input_tokens"],
                            "output_tokens": response["output_tokens"], "cost": response["cost"], "response_id": response["response_id"]})
                else:
                    usage["failed_attempts"] += 1
                    failure = response.get("error") or "request_failed"
                with (out / "api_attempts.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"trial": number, "mechanism_id": item[0], "sample_token": _hash(row.sample_id)[:16],
                                "request_hash": request_hash, "status": response.get("status"), "model": response.get("model"),
                                "provider": response.get("provider"), "input_tokens": response.get("input_tokens"),
                                "output_tokens": response.get("output_tokens"), "cost": response.get("cost"),
                                "success": response.get("success"), "failure": response.get("error")}, sort_keys=True) + "\n")
                _write(out / "usage_summary.json", usage)
                if not response["success"] or usage["reported_cost_usd"] > 0.05:
                    failure = failure or "Reported-cost cap reached."
                    break
            scores = response["scores"]
            paired.append({"sample_id": row.sample_id, "target": target, "band": row.region, "group_id": row.group_id,
                           "v4_score": scores["v4"] / 2, "candidate_score": scores["candidate"] / 2,
                           "request_hash": request_hash, "model": response["model"], "provider": response["provider"],
                           "input_tokens": response["input_tokens"], "output_tokens": response["output_tokens"],
                           "reported_cost_usd": response["cost"], "cache_hit": bool(cached)})
    pd.DataFrame(paired).to_csv(out / "responses" / f"trial_{number:02d}.csv", index=False)
    result = validate_candidate(paired, _read(out / "group_split.json"), validation_spec)
    if failure:
        result.update({"decision": "INCONCLUSIVE", "status": "INCONCLUSIVE", "reasons": [failure]})
    result.update({"trial": number, "mechanism_id": item[0], "changed_element": item[1],
                   "api_requests_cumulative": usage["api_requests"], "reported_cost_usd_cumulative": usage["reported_cost_usd"],
                   "api_responses_this_trial": len(paired), "expected_paired_samples": len(plan),
                   "selection_bias": "Same development windows reused across trials."})
    _write(result_path, result)
    _write(out / "usage_summary.json", usage)
    for registry_row in registry:
        if registry_row["mechanism_id"] == item[0]:
            registry_row["status"] = result["decision"]
            break
    _csv(registry_path, registry, list(registry[0].keys()))
    metrics, checks = result.get("metrics"), result.get("checks")
    if metrics:
        v, c, d = metrics["v4"], metrics["candidate"], result["metric_differences"]["overall"]
        auc, f1 = f"{c['macro_roc_auc']:.4f} / {v['macro_roc_auc']:.4f}", f"{c['oof_macro_f1']:.4f} / {v['oof_macro_f1']:.4f}"
        danger = f"normal->high rate {d['normal_to_high_rate']:+.3f}; near->low count {d['near_failure_to_low_count']:+d}"
        branches = [checks["roc_branch"]["passed"], checks["f1_branch"]["passed"], checks["safety"]["passed"]]
        summary = {"trial": number, "mechanism_id": item[0], "decision": result["decision"], "sample_count": len(paired),
                   "group_count": len(set(x["group_id"] for x in paired)), "candidate_macro_roc_auc": c["macro_roc_auc"],
                   "v4_macro_roc_auc": v["macro_roc_auc"], "delta_macro_roc_auc": d["macro_roc_auc"],
                   "candidate_oof_macro_f1": c["oof_macro_f1"], "v4_oof_macro_f1": v["oof_macro_f1"],
                   "delta_oof_macro_f1": d["oof_macro_f1"], "normal_to_high_rate_delta": d["normal_to_high_rate"],
                   "near_failure_to_low_count_delta": d["near_failure_to_low_count"], "roc_branch": branches[0],
                   "f1_branch": branches[1], "safety": branches[2], "api_requests_cumulative": usage["api_requests"],
                   "reported_cost_usd_cumulative": usage["reported_cost_usd"], "reason": " ".join(result["reasons"])}
    else:
        auc = f1 = "INCONCLUSIVE / INCONCLUSIVE"
        danger = "unavailable"
        branches = ["INCONCLUSIVE"] * 3
        summary = {"trial": number, "mechanism_id": item[0], "decision": "INCONCLUSIVE", "sample_count": len(paired),
                   "group_count": len(set(x["group_id"] for x in paired)), "api_requests_cumulative": usage["api_requests"],
                   "reported_cost_usd_cumulative": usage["reported_cost_usd"], "reason": " ".join(result["reasons"])}
    path = out / "trial_summary.csv"
    previous = list(csv.DictReader(path.open(encoding="utf-8", newline=""))) if path.exists() else []
    previous = [x for x in previous if int(x["trial"]) != number]
    previous.append(summary)
    _csv(path, previous, ["trial", "mechanism_id", "decision", "sample_count", "group_count", "candidate_macro_roc_auc",
         "v4_macro_roc_auc", "delta_macro_roc_auc", "candidate_oof_macro_f1", "v4_oof_macro_f1", "delta_oof_macro_f1",
         "normal_to_high_rate_delta", "near_failure_to_low_count_delta", "roc_branch", "f1_branch", "safety",
         "api_requests_cumulative", "reported_cost_usd_cumulative", "reason"])
    table = f"| {number} | {item[0]} | {item[1]} | {len(paired)}/{len(set(x['group_id'] for x in paired))} | {auc} | {f1} | {danger} | {branches[0]} | {branches[1]} | {branches[2]} | {result['decision']} | {usage['api_requests']} / USD {usage['reported_cost_usd']:.6f} |"
    with (out / "trial_log.md").open("a", encoding="utf-8") as handle:
        handle.write(table + "\n\n" + f"Trial {number}: {' '.join(result['reasons'])} Reused development windows make this exploratory and selection-biased.\n")
    if number == 1:
        costs = []
        attempts = out / "api_attempts.jsonl"
        if attempts.exists():
            costs = [float(x["cost"]) for x in map(json.loads, attempts.read_text(encoding="utf-8").splitlines())
                     if x.get("trial") == 1 and x.get("cost") is not None]
        avg = float(np.mean(costs)) if costs else None
        max_cost = max(costs) if costs else None
        unit = max(0.000075, (max_cost or 0) * 1.5, (avg or 0) * 1.5)
        future_dev, final_reserve = 9 * len(plan), len(plan)
        retries = min(105, int(math.ceil(0.1 * (future_dev + final_reserve))))
        projected_calls = usage["api_requests"] + future_dev + final_reserve + retries
        projected_cost = usage["reported_cost_usd"] + unit * (future_dev + final_reserve + retries)
        possible_trials = max(0, int((0.05 - usage["reported_cost_usd"] - unit * (final_reserve + retries)) / (unit * len(plan))))
        forecast = {"pilot_requests": usage["api_requests"], "pilot_success_rows": len(paired),
                    "pilot_average_cost_usd": avg, "pilot_max_cost_usd": max_cost, "unit_cost_reserve_usd": unit,
                    "remaining_trials_planned": 9, "hypothetical_final_requests": final_reserve,
                    "retry_reserve_requests": retries, "projected_total_requests": projected_calls,
                    "projected_total_cost_usd": projected_cost, "recommended_max_trial": min(10, 1 + possible_trials),
                    "model_provider_parse_usage_verified": len(paired) == len(plan) and bool(costs)}
        _write(out / "budget_forecast_after_pilot.json", forecast)
        with (out / "trial_log.md").open("a", encoding="utf-8") as handle:
            handle.write("\nPilot budget forecast: " + json.dumps(forecast, sort_keys=True) + "\n")
    print(json.dumps({"trial": number, "mechanism_id": item[0], "decision": result["decision"],
          "metrics": metrics, "checks": checks, "reasons": result["reasons"],
          "api_requests_cumulative": usage["api_requests"], "reported_cost_usd_cumulative": usage["reported_cost_usd"]}, sort_keys=True))
    return result


def finalize_experiment(root=ROOT):
    root = Path(root).resolve()
    out = root / "artifacts/jev_v4_edge_trials"
    _frozen(root, out)
    feasibility = _read(root / "artifacts/jev_prompt_optimization/noul_followup/feasibility_status.json")
    path = out / "trial_summary.csv"
    trials = list(csv.DictReader(path.open(encoding="utf-8", newline=""))) if path.exists() else []
    keeps = [x for x in trials if x.get("decision") == "PROVISIONAL_KEEP"]
    selected = sorted(keeps, key=lambda x: (-float(x["candidate_macro_roc_auc"]), -float(x["candidate_oof_macro_f1"]), int(x["trial"])))[0] if keeps else None
    result = {"status": "INCONCLUSIVE", "verified": False, "selected_development_candidate": selected,
              "unused_required_near_failure_groups": int(feasibility.get("leak_near_failure", {}).get("unused_groups", 0)),
              "reason": feasibility.get("reason"), "primary_metric": "macro_roc_auc", "group_paired_bootstrap": None,
              "bootstrap_reason": "No untouched validation sample can provide all target x TTF-band classes.",
              "selection_bias": "Same development windows reused across trials.",
              "limitations": ["Previously observed 05b/Noul groups are not independent validation.",
                              "TTF bands are surrogate labels, not current-state ground truth.",
                              "Balanced-sample precision and PR-AUC are not operating prevalence."]}
    _write(out / "final_confirmation.json", result)
    return result


def reevaluate_saved_trials(root=ROOT):
    """Version the numeric boundary fix and replay every saved pair without API calls."""
    root = Path(root).resolve()
    out = root / "artifacts/jev_v4_edge_trials"
    old_manifest = _read(out / "frozen_manifest.json")
    old_spec = _read(out / "validation_spec.json")
    usage = _read(out / "usage_summary.json")
    previous_spec_hash = old_manifest.get("spec_sha256")
    previous_validator_hash = old_manifest.get("validator_sha256")
    if usage.get("api_requests", 0) <= 0:
        raise RuntimeError("No saved API trial exists to re-evaluate.")
    if old_spec.get("spec_version") != "1.1":
        raise RuntimeError("Expected the frozen v1.1 spec before migration.")
    if previous_validator_hash == _hash((root / "src/fabjudge/jev_v4_validation.py").read_bytes()):
        raise RuntimeError("The v1.2 validator source must differ from the frozen v1.1 validator.")

    spec = _spec()
    spec_bytes = (json.dumps(spec, sort_keys=True, indent=2) + "\n").encode()
    spec_hash = _hash(spec_bytes)
    split = _read(out / "group_split.json")
    split_hash = _hash((out / "group_split.json").read_bytes())
    plan_hash = _hash((out / "development_sample_plan.csv").read_bytes())
    questions_hash = _hash((out / "candidate_questions.json").read_bytes())
    new_validator_hash = _hash((root / "src/fabjudge/jev_v4_validation.py").read_bytes())
    new_runner_hash = _hash(Path(__file__).read_bytes())

    all_results = pd.read_csv(root / "artifacts/jev_prompt_optimization/all_results.csv",
                              usecols=["sample_id", "prompt_id", "risk_score", "subset", "target", "region", "group_id"])
    plan = pd.read_csv(out / "development_sample_plan.csv")
    old = all_results.loc[(all_results.subset == "development") & all_results.sample_id.isin(plan.sample_id)]
    preflight_pairs = old.pivot(index=["sample_id", "target", "region", "group_id"],
                                columns="prompt_id", values="risk_score").reset_index()
    preflight_pairs = preflight_pairs.rename(columns={"region": "band", "v4": "v4_score", "v5": "candidate_score"})
    preflight = validate_candidate(preflight_pairs.to_dict("records"), split, spec)
    if preflight["decision"] == "INCONCLUSIVE":
        raise RuntimeError("Saved-response preflight inconclusive under v1.2: " + str(preflight["reasons"]))

    saved = []
    index = 1
    while index <= 10:
        response_path = out / "responses" / f"trial_{index:02d}.csv"
        result_path = out / "validations" / f"trial_{index:02d}.json"
        if not response_path.exists() and not result_path.exists():
            break
        if not response_path.exists() or not result_path.exists():
            raise RuntimeError(f"Saved response/result pair is incomplete for trial {index}.")
        previous = _read(result_path)
        response_rows = pd.read_csv(response_path).to_dict("records")
        recalculated = validate_candidate(response_rows, split, spec)
        recalculated.update({k: previous[k] for k in ("trial", "mechanism_id", "changed_element",
                              "api_requests_cumulative", "reported_cost_usd_cumulative",
                              "api_responses_this_trial", "expected_paired_samples", "selection_bias") if k in previous})
        recalculated["validation_version"] = spec["spec_version"]
        recalculated["validation_spec_sha256"] = spec_hash
        recalculated["previous_validation"] = {"spec_version": old_spec["spec_version"],
                    "spec_sha256": old_manifest["spec_sha256"], "validator_sha256": old_manifest["validator_sha256"],
                    "decision": previous.get("decision")}
        saved.append(recalculated)
        print(f"Re-evaluated trial {index}: {previous.get('decision')} -> {recalculated.get('decision')}; "
              f"v4 macro ROC-AUC={((recalculated.get('metrics') or {}).get('v4') or {}).get('macro_roc_auc')}", flush=True)
        index += 1
    if not saved:
        raise RuntimeError("No complete saved trials found.")

    old_log_path = out / "trial_log.md"
    revisions = out / "revisions"
    revisions.mkdir(parents=True, exist_ok=True)
    old_log_copy = revisions / "trial_log_v1_1_before_numeric_fix.md"
    if not old_log_copy.exists():
        old_log_copy.write_text(old_log_path.read_text(encoding="utf-8"), encoding="utf-8")

    (out / "validation_spec.json").write_bytes(spec_bytes)
    (out / "validation_spec.sha256").write_text(spec_hash + "  validation_spec.json\n", encoding="ascii")
    _write(out / "validator_preflight_old_05b.json", {
        "use": "offline implementation preflight only; not a trial", "decision": preflight["decision"],
        "spec_version": spec["spec_version"], "spec_sha256": spec_hash,
        "sample_count": preflight["sample_count"], "metrics": preflight["metrics"], "checks": preflight["checks"]})

    summary_rows = []
    registry_path = out / "hypothesis_registry.csv"
    registry = list(csv.DictReader(registry_path.open(encoding="utf-8", newline="")))
    for result in saved:
        _write(out / "validations" / f"trial_{int(result['trial']):02d}.json", result)
        metrics, checks = result.get("metrics"), result.get("checks")
        row = {"trial": result["trial"], "mechanism_id": result["mechanism_id"],
               "decision": result["decision"], "sample_count": result["sample_count"],
               "group_count": result["group_count"], "reason": " ".join(result["reasons"]),
               "api_requests_cumulative": result["api_requests_cumulative"],
               "reported_cost_usd_cumulative": result["reported_cost_usd_cumulative"]}
        if metrics:
            candidate, v4 = metrics["candidate"], metrics["v4"]
            delta = result["metric_differences"]["overall"]
            row.update({"candidate_macro_roc_auc": candidate["macro_roc_auc"], "v4_macro_roc_auc": v4["macro_roc_auc"],
                "delta_macro_roc_auc": delta["macro_roc_auc"], "candidate_oof_macro_f1": candidate["oof_macro_f1"],
                "v4_oof_macro_f1": v4["oof_macro_f1"], "delta_oof_macro_f1": delta["oof_macro_f1"],
                "normal_to_high_rate_delta": delta["normal_to_high_rate"],
                "near_failure_to_low_count_delta": delta["near_failure_to_low_count"],
                "roc_branch": checks["roc_branch"]["passed"], "f1_branch": checks["f1_branch"]["passed"],
                "safety": checks["safety"]["passed"]})
        summary_rows.append(row)
        for registry_row in registry:
            if registry_row["mechanism_id"] == result["mechanism_id"]:
                registry_row["status"] = result["decision"]
                break
    _csv(out / "trial_summary.csv", summary_rows, ["trial", "mechanism_id", "decision", "sample_count", "group_count",
        "candidate_macro_roc_auc", "v4_macro_roc_auc", "delta_macro_roc_auc", "candidate_oof_macro_f1", "v4_oof_macro_f1",
        "delta_oof_macro_f1", "normal_to_high_rate_delta", "near_failure_to_low_count_delta", "roc_branch", "f1_branch",
        "safety", "api_requests_cumulative", "reported_cost_usd_cumulative", "reason"])
    _csv(registry_path, registry, list(registry[0].keys()))

    lines = ["# JEV v4 Score edge trials", "", f"- Frozen validation spec v{spec['spec_version']} SHA256: {spec_hash}",
        f"- Frozen validator SHA256: {new_validator_hash}",
        "- v1.2 migration: inclusive float threshold comparisons use epsilon 1e-12; metric definitions, cutoffs, samples and group folds are unchanged.",
        "- v1.1 results were replayed from every saved paired response through the current trial; exact v4 answers were re-evaluated alongside each candidate.",
        f"- Offline old-05b development preflight under v1.2: {preflight['decision']} (implementation check only, not selection).",
        "- 05b and prior artifacts are read-only inputs.", f"- Development: {len(plan)} paired windows / {plan.group_id.nunique()} groups.",
        "- Fixed 3-fold StratifiedGroupKFold, seed 42; fold training groups alone fit thresholds.",
        "- New-call cap: 600 requests and USD 0.05, separate from historical usage.",
        "- Final independent groups: INCONCLUSIVE; unused leak/near-failure groups are zero.",
        "- Reusing development windows across trials is selection-biased.", "",
        "| trial | new mechanism_id | changed element | sample/groups | candidate/v4 ROC-AUC | candidate/v4 OOF macro-F1 | dangerous error differences | roc_branch | f1_branch | safety | validator verdict | cumulative calls/cost |",
        "|---:|---|---|---:|---:|---:|---|---|---|---|---|---|"]
    for result in saved:
        m, d, checks = result.get("metrics"), result.get("metric_differences"), result.get("checks")
        if m:
            c, v, delta = m["candidate"], m["v4"], d["overall"]
            auc = f"{c['macro_roc_auc']:.4f} / {v['macro_roc_auc']:.4f} (Δ{delta['macro_roc_auc']:+.4f})"
            f1 = f"{c['oof_macro_f1']:.4f} / {v['oof_macro_f1']:.4f} (Δ{delta['oof_macro_f1']:+.4f})"
            danger = f"normal->high rate {delta['normal_to_high_rate']:+.3f}; near->low count {delta['near_failure_to_low_count']:+d}"
            branches = [checks["roc_branch"]["passed"], checks["f1_branch"]["passed"], checks["safety"]["passed"]]
        else:
            auc = f1 = "INCONCLUSIVE / INCONCLUSIVE"
            danger, branches = "unavailable", ["INCONCLUSIVE"] * 3
        lines.append(f"| {result['trial']} | {result['mechanism_id']} | {result.get('changed_element','')} | {result.get('sample_count',0)}/{result.get('group_count',0)} | {auc} | {f1} | {danger} | {branches[0]} | {branches[1]} | {branches[2]} | {result['decision']} | {result.get('api_requests_cumulative',0)} / USD {result.get('reported_cost_usd_cumulative',0):.6f} |")
        lines.append("")
        lines.append(f"Trial {result['trial']}: {' '.join(result['reasons'])} Reused development windows make this exploratory and selection-biased.")
    forecast_path = out / "budget_forecast_after_pilot.json"
    if forecast_path.exists():
        lines.extend(["", "Pilot budget forecast: " + json.dumps(_read(forecast_path), sort_keys=True)])
    old_manifest.update({"spec_sha256": spec_hash, "validator_sha256": new_validator_hash,
        "runner_sha256": new_runner_hash, "split_sha256": split_hash, "sample_sha256": plan_hash,
        "questions_sha256": questions_hash})
    old_manifest["validation_revision"] = {"from_spec_version": old_spec["spec_version"], "to_spec_version": spec["spec_version"],
        "previous_spec_sha256": _hash((json.dumps(old_spec, sort_keys=True, indent=2) + "\n").encode()),
        "reason": "Float boundary comparison correction; stored rounds replayed before further API calls."}
    _write(out / "frozen_manifest.json", old_manifest)
    _write(out / "validator_migration_v1_2.json", {
        "reason": "Inclusive floating-point threshold equality was misclassified at exact ROC-AUC delta 0.02.",
        "spec_version_from": old_spec["spec_version"], "spec_version_to": spec["spec_version"],
        "previous_spec_sha256": previous_spec_hash,
        "new_spec_sha256": spec_hash, "previous_validator_sha256": previous_validator_hash,
        "new_validator_sha256": new_validator_hash, "reevaluated_trials": [x["trial"] for x in saved],
        "decisions": [{"trial": x["trial"], "previous": x["previous_validation"]["decision"], "current": x["decision"],
                       "roc_branch": (x.get("checks") or {}).get("roc_branch", {}).get("passed")} for x in saved],
        "prior_trial_log_backup": str(old_log_copy.relative_to(root)), "api_requests_added": 0})
    old_log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"spec_version": spec["spec_version"], "spec_sha256": spec_hash,
            "validator_sha256": new_validator_hash, "reevaluated_trials": len(saved),
            "decisions": [{"trial": x["trial"], "decision": x["decision"],
                          "roc_branch": (x.get("checks") or {}).get("roc_branch", {}).get("passed")} for x in saved],
            "old_05b_preflight": preflight["decision"], "api_requests_added": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--trial", type=int)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--reevaluate-saved", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare_experiment(ROOT), sort_keys=True))
    elif args.trial:
        run_trial(args.trial, ROOT)
    elif args.finalize:
        print(json.dumps(finalize_experiment(ROOT), sort_keys=True))
    elif args.reevaluate_saved:
        print(json.dumps(reevaluate_saved_trials(ROOT), sort_keys=True))
    else:
        parser.error("Choose --prepare, --trial N, or --finalize.")


if __name__ == "__main__":
    main()
