from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

LABELS = ("normal", "boundary", "near_failure")
PROMPTS = ("v4", "candidate")


class InconclusiveError(ValueError):
    pass


def _thresholds(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if set(labels.tolist()) != {0, 1, 2}:
        raise InconclusiveError("A fold training target is missing a TTF class.")
    unique = sorted(set(scores.tolist()))
    values = sorted({0.0, 1.0} | {(a + b) / 2 for a, b in zip(unique[:-1], unique[1:])})
    best, pair = None, None
    for t1, t2 in combinations(values, 2):
        pred = np.where(scores <= t1, 0, np.where(scores <= t2, 1, 2))
        value = float(f1_score(labels, pred, labels=[0, 1, 2], average="macro", zero_division=0))
        if best is None or value > best + 1e-12 or (abs(value - best) <= 1e-12 and (t1, t2) < pair):
            best, pair = value, (float(t1), float(t2))
    if pair is None:
        raise InconclusiveError("No ordered threshold pair can be selected.")
    return {"t1": pair[0], "t2": pair[1], "training_macro_f1": best}


def _validate_folds(frame, split, spec):
    ids = set(frame.sample_id.astype(str))
    groups = dict(zip(frame.sample_id.astype(str), frame.group_id.astype(str)))
    if not isinstance(split, dict) or len(split.get("folds", [])) != int(spec["threshold_cv"]["n_splits"]):
        raise InconclusiveError("The frozen group split is missing or has the wrong fold count.")
    folds, seen = [], []
    for i, fold in enumerate(split["folds"]):
        train = set(map(str, fold.get("train_sample_ids", [])))
        valid = set(map(str, fold.get("validation_sample_ids", [])))
        if not train or not valid or train & valid or train | valid != ids:
            raise InconclusiveError(f"Invalid sample partition in fold {i}.")
        if {groups[x] for x in train} & {groups[x] for x in valid}:
            raise InconclusiveError(f"Group leakage in fold {i}.")
        folds.append({"fold": int(fold.get("fold", i)),
                      "train_sample_ids": sorted(train),
                      "validation_sample_ids": sorted(valid)})
        seen.extend(valid)
    if set(seen) != ids or len(seen) != len(set(seen)):
        raise InconclusiveError("Every sample must receive exactly one OOF prediction.")
    return folds


def _oof(frame, folds, spec, prompt):
    col = prompt + "_score"
    lookup = frame.set_index("sample_id", drop=False)
    labels = frame.set_index("sample_id").band.map(spec["class_mapping"]).astype(int).to_dict()
    predictions, thresholds = {}, []
    for fold in folds:
        cuts = {}
        for target in spec["targets"]:
            train = lookup.loc[fold["train_sample_ids"]]
            train = train.loc[train.target.astype(str) == target]
            y = np.asarray([labels[str(x)] for x in train.sample_id], dtype=int)
            cuts[target] = _thresholds(train[col].to_numpy(float), y)
        valid = lookup.loc[fold["validation_sample_ids"]]
        for row in valid.itertuples():
            cut = cuts[str(row.target)]
            score = float(getattr(row, col))
            predictions[str(row.sample_id)] = 0 if score <= cut["t1"] else (1 if score <= cut["t2"] else 2)
        thresholds.append({"prompt": prompt, "fold": fold["fold"], "by_target": cuts})
    if set(predictions) != set(frame.sample_id.astype(str)):
        raise InconclusiveError("Incomplete OOF predictions.")
    return predictions, thresholds


def _metrics(frame, predictions, spec, score_col):
    by_target = {}
    nh = nl = normals = nears = 0
    for target in spec["targets"]:
        rows = frame.loc[frame.target.astype(str) == target].copy()
        y = rows.band.map(spec["class_mapping"]).astype(int).to_numpy()
        pred = np.asarray([predictions[str(x)] for x in rows.sample_id.astype(str)], dtype=int)
        if set(y.tolist()) != {0, 1, 2}:
            raise InconclusiveError(f"An evaluation target is missing a TTF class: {target}.")
        keep = y != 1
        binary = (y[keep] == 2).astype(int)
        if set(binary.tolist()) != {0, 1}:
            raise InconclusiveError(f"ROC-AUC needs normal and near_failure for {target}.")
        auc = float(roc_auc_score(binary, rows[score_col].to_numpy(float)[keep]))
        mf1 = float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0))
        nmask, fmask = y == 0, y == 2
        nbad, fbad = int(np.sum(nmask & (pred == 2))), int(np.sum(fmask & (pred == 0)))
        n, f = int(nmask.sum()), int(fmask.sum())
        nh += nbad
        nl += fbad
        normals += n
        nears += f
        by_target[target] = {
            "roc_auc_normal_vs_near_failure": auc,
            "oof_macro_f1": mf1,
            "dangerous_errors": {
                "normal_to_high": {"count": nbad, "denominator": n, "rate": nbad / n},
                "near_failure_to_low": {"count": fbad, "denominator": f, "rate": fbad / f},
            },
        }
    return {
        "per_target": by_target,
        "macro_roc_auc": float(np.mean([by_target[t]["roc_auc_normal_vs_near_failure"] for t in spec["targets"]])),
        "oof_macro_f1": float(np.mean([by_target[t]["oof_macro_f1"] for t in spec["targets"]])),
        "dangerous_errors": {
            "normal_to_high": {"count": nh, "denominator": normals, "rate": nh / normals},
            "near_failure_to_low": {"count": nl, "denominator": nears, "rate": nl / nears},
        },
    }


def _evaluate(frame, folds, spec):
    metrics, threshold_rows = {}, {}
    for prompt in PROMPTS:
        pred, threshold_rows[prompt] = _oof(frame, folds, spec, prompt)
        metrics[prompt] = _metrics(frame, pred, spec, prompt + "_score")
    return metrics, threshold_rows


def _delta(candidate, baseline):
    per_target = {}
    for target in candidate["per_target"]:
        c, b = candidate["per_target"][target], baseline["per_target"][target]
        per_target[target] = {
            "roc_auc": c["roc_auc_normal_vs_near_failure"] - b["roc_auc_normal_vs_near_failure"],
            "oof_macro_f1": c["oof_macro_f1"] - b["oof_macro_f1"],
            "normal_to_high_count": c["dangerous_errors"]["normal_to_high"]["count"] - b["dangerous_errors"]["normal_to_high"]["count"],
            "normal_to_high_rate": c["dangerous_errors"]["normal_to_high"]["rate"] - b["dangerous_errors"]["normal_to_high"]["rate"],
            "near_failure_to_low_count": c["dangerous_errors"]["near_failure_to_low"]["count"] - b["dangerous_errors"]["near_failure_to_low"]["count"],
            "near_failure_to_low_rate": c["dangerous_errors"]["near_failure_to_low"]["rate"] - b["dangerous_errors"]["near_failure_to_low"]["rate"],
        }
    return {
        "overall": {
            "macro_roc_auc": candidate["macro_roc_auc"] - baseline["macro_roc_auc"],
            "oof_macro_f1": candidate["oof_macro_f1"] - baseline["oof_macro_f1"],
            "normal_to_high_rate": candidate["dangerous_errors"]["normal_to_high"]["rate"] - baseline["dangerous_errors"]["normal_to_high"]["rate"],
            "normal_to_high_count": candidate["dangerous_errors"]["normal_to_high"]["count"] - baseline["dangerous_errors"]["normal_to_high"]["count"],
            "near_failure_to_low_count": candidate["dangerous_errors"]["near_failure_to_low"]["count"] - baseline["dangerous_errors"]["near_failure_to_low"]["count"],
            "near_failure_to_low_rate": candidate["dangerous_errors"]["near_failure_to_low"]["rate"] - baseline["dangerous_errors"]["near_failure_to_low"]["rate"],
        },
        "per_target": per_target,
    }


def _checks(candidate, baseline, spec):
    rule = spec["provisional_keep_rule"]
    eps = float(spec["numeric_comparison"]["inclusive_float_epsilon"])
    roc = {
        "candidate_macro_roc_auc_ge_0_62": candidate["macro_roc_auc"] >= rule["roc_candidate_min"] - eps,
        "candidate_minus_v4_macro_roc_auc_ge_0_02": candidate["macro_roc_auc"] - baseline["macro_roc_auc"] >= rule["roc_delta_min"] - eps,
    }
    f1 = {
        "candidate_oof_macro_f1_ge_0_45": candidate["oof_macro_f1"] >= rule["f1_candidate_min"] - eps,
        "candidate_minus_v4_oof_macro_f1_ge_0_03": candidate["oof_macro_f1"] - baseline["oof_macro_f1"] >= rule["f1_delta_min"] - eps,
    }
    safety = {
        "candidate_roc_auc_ge_v4_minus_0_01": candidate["macro_roc_auc"] >= baseline["macro_roc_auc"] - rule["safety_roc_tolerance"] - eps,
        "candidate_macro_f1_ge_v4_minus_0_01": candidate["oof_macro_f1"] >= baseline["oof_macro_f1"] - rule["safety_f1_tolerance"] - eps,
        "candidate_near_failure_to_low_count_le_v4": candidate["dangerous_errors"]["near_failure_to_low"]["count"] <= baseline["dangerous_errors"]["near_failure_to_low"]["count"],
        "candidate_normal_to_high_rate_le_v4_plus_0_05": candidate["dangerous_errors"]["normal_to_high"]["rate"] <= baseline["dangerous_errors"]["normal_to_high"]["rate"] + rule["normal_to_high_rate_tolerance"] + eps,
    }
    return {"roc_branch": {"passed": all(roc.values()), "conditions": roc},
            "f1_branch": {"passed": all(f1.values()), "conditions": f1},
            "safety": {"passed": all(safety.values()), "conditions": safety}}


def validate_candidate(paired_responses, group_split, spec):
    try:
        frame = pd.DataFrame(paired_responses).copy()
        required = {"sample_id", "target", "band", "group_id", "v4_score", "candidate_score"}
        missing = required - set(frame.columns)
        if missing:
            raise InconclusiveError("Missing paired fields: " + ", ".join(sorted(missing)))
        if frame.empty or frame.sample_id.astype(str).duplicated().any():
            raise InconclusiveError("Empty or duplicated paired samples.")
        for col in ("sample_id", "target", "band", "group_id"):
            frame[col] = frame[col].astype(str)
        if set(frame.target) != set(spec["targets"]) or set(frame.band) != set(LABELS):
            raise InconclusiveError("Frozen target set or all three TTF bands are not present.")
        for col in ("v4_score", "candidate_score"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
            if frame[col].isna().any() or not np.isfinite(frame[col].to_numpy(float)).all() or not frame[col].between(0, 1).all():
                raise InconclusiveError(col + " must be finite native Score / 2 in [0,1].")
        for target in spec["targets"]:
            if set(frame.loc[frame.target == target, "band"]) != set(LABELS):
                raise InconclusiveError("A target is missing a TTF band: " + target)
        folds = _validate_folds(frame, group_split, spec)
        metrics, thresholds = _evaluate(frame, folds, spec)
        v4, candidate = metrics["v4"], metrics["candidate"]
        deltas, checks = _delta(candidate, v4), _checks(candidate, v4, spec)
        reasons = []
        if not checks["roc_branch"]["passed"]:
            reasons.append("roc_branch is false.")
        if not checks["f1_branch"]["passed"]:
            reasons.append("f1_branch is false.")
        if not checks["safety"]["passed"]:
            reasons.append("Safety failed: " + ", ".join(k for k, v in checks["safety"]["conditions"].items() if not v) + ".")
        decision = "PROVISIONAL_KEEP" if (checks["roc_branch"]["passed"] or checks["f1_branch"]["passed"]) and checks["safety"]["passed"] else "REJECT"
        group_map = dict(zip(frame.sample_id, frame.group_id))
        influence = []
        for group in sorted(frame.group_id.unique()):
            reduced = frame.loc[frame.group_id != group].copy()
            reduced_folds = [{
                "fold": f["fold"],
                "train_sample_ids": [x for x in f["train_sample_ids"] if group_map[x] != group],
                "validation_sample_ids": [x for x in f["validation_sample_ids"] if group_map[x] != group],
            } for f in folds]
            try:
                mm, _ = _evaluate(reduced, reduced_folds, spec)
                dd = _delta(mm["candidate"], mm["v4"])
                influence.append({
                    "excluded_group": group, "status": "CALCULATED",
                    "candidate_macro_roc_auc_change": mm["candidate"]["macro_roc_auc"] - candidate["macro_roc_auc"],
                    "v4_macro_roc_auc_change": mm["v4"]["macro_roc_auc"] - v4["macro_roc_auc"],
                    "candidate_minus_v4_macro_roc_auc_change": dd["overall"]["macro_roc_auc"] - deltas["overall"]["macro_roc_auc"],
                    "candidate_oof_macro_f1_change": mm["candidate"]["oof_macro_f1"] - candidate["oof_macro_f1"],
                    "v4_oof_macro_f1_change": mm["v4"]["oof_macro_f1"] - v4["oof_macro_f1"],
                    "candidate_minus_v4_oof_macro_f1_change": dd["overall"]["oof_macro_f1"] - deltas["overall"]["oof_macro_f1"],
                    "candidate_normal_to_high_rate_change": mm["candidate"]["dangerous_errors"]["normal_to_high"]["rate"] - candidate["dangerous_errors"]["normal_to_high"]["rate"],
                    "v4_normal_to_high_rate_change": mm["v4"]["dangerous_errors"]["normal_to_high"]["rate"] - v4["dangerous_errors"]["normal_to_high"]["rate"],
                    "candidate_near_failure_to_low_count_change": mm["candidate"]["dangerous_errors"]["near_failure_to_low"]["count"] - candidate["dangerous_errors"]["near_failure_to_low"]["count"],
                    "v4_near_failure_to_low_count_change": mm["v4"]["dangerous_errors"]["near_failure_to_low"]["count"] - v4["dangerous_errors"]["near_failure_to_low"]["count"],
                })
            except (InconclusiveError, KeyError, ValueError) as exc:
                influence.append({"excluded_group": group, "status": "INCONCLUSIVE", "reason": str(exc)})
        return {"status": decision, "decision": decision, "reasons": reasons or ["Development-only provisional rule passed."],
                "sample_count": len(frame), "group_count": int(frame.group_id.nunique()), "metrics": metrics,
                "metric_differences": deltas, "checks": checks, "fold_thresholds": thresholds,
                "group_sensitivity": influence}
    except (InconclusiveError, KeyError, TypeError, ValueError) as exc:
        frame = pd.DataFrame(paired_responses)
        return {"status": "INCONCLUSIVE", "decision": "INCONCLUSIVE", "reasons": [str(exc)],
                "sample_count": len(frame), "group_count": int(frame.group_id.nunique()) if "group_id" in frame else None,
                "metrics": None, "metric_differences": None, "checks": None,
                "fold_thresholds": None, "group_sensitivity": None}
