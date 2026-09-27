"""Offline audits and exploratory score thresholds for the completed 05b run.

Only saved responses and processed window features are read. The module never
loads credentials or makes network requests. Outputs contain aggregates and
local identifiers, never the raw 20-feature input vectors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold


REGIONS = ("normal", "boundary", "near_failure")
PROMPTS = ("v4", "v5")
MODEL = "typesafe/jev-1.13"
PHYSICAL = ("source_file", "sequence_index", "window_index")
PAIR_KEY = ("sample_id", "prompt_id")


def _required(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"05b offline diagnostics needs this saved file: {path}")
    return path


def _json(path: Path):
    return json.loads(_required(path).read_text(encoding="utf-8"))


def _write_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _physical_keys(frame: pd.DataFrame) -> set[tuple[str, int, int]]:
    return {(str(x.source_file), int(x.sequence_index), int(x.window_index))
            for x in frame.loc[:, list(PHYSICAL)].itertuples(index=False)}


def _check_inputs(root: Path):
    out = root / "artifacts" / "jev_prompt_optimization"
    config = _json(out / "config.json")
    selection = _json(out / "development_selection.json")
    questions = _json(out / "decision_questions.json")
    request_audit = _json(out / "request_body_audit.json")
    stats = _json(root / "artifacts" / "jev_tut" / "train_scaler_stats.json")
    base_config = _json(root / "artifacts" / "jev_tut" / "config.json")
    sample = pd.read_csv(_required(out / "sample_audit.csv"))
    cols = ["sample_id", "prompt_id", "subset", "risk_score", "native_score", "confidence",
            "request_hash", "model", "provider", "success", "parse_success", "usage_valid"]
    results = pd.read_csv(_required(out / "all_results.csv"), usecols=cols)
    repeats = pd.read_csv(_required(out / "repeatability.csv"),
                          usecols=["sample_id", "prompt_id", "baseline_risk", "repeat_risk",
                                   "risk_delta_repeat_minus_primary"])
    old = pd.read_csv(_required(root / "artifacts" / "jev_tut" / "selected_samples.csv"))
    old_result = pd.read_csv(_required(root / "artifacts" / "jev_tut" / "robust_results.csv"),
                             usecols=["sample_id", "target", "region", "risk_score"])
    macro = pd.read_csv(_required(out / "macro_comparison.csv"))
    features = base_config["feature_columns"]
    targets = base_config["target_columns"]

    if config["sample_count"] != 270 or len(sample) != 270 or len(results) != 540:
        raise ValueError("Expected the completed 270-sample / 540-primary-response 05b run.")
    if set(sample.subset) != {"development", "holdout"} or set(sample.region) != set(REGIONS):
        raise ValueError("Unexpected saved sample subsets or TTF bands.")
    if sample.sample_id.duplicated().any() or results.duplicated(list(PAIR_KEY)).any():
        raise ValueError("Saved primary sample or response keys are duplicated.")
    if set(results.prompt_id) != set(PROMPTS) or set(results.sample_id) != set(sample.sample_id):
        raise ValueError("Every selected sample needs one v4 and one v5 primary response.")
    if not results.loc[:, ["success", "parse_success", "usage_valid"]].all().all():
        raise ValueError("Saved 05b primary responses include failed parsing or usage checks.")
    if not np.allclose(results.risk_score, results.native_score / 2, atol=1e-12, rtol=0):
        raise ValueError("Saved risk_score differs from native Jev score / 2.")
    if results.risk_score.isna().any() or not results.risk_score.between(0, 1).all():
        raise ValueError("Invalid or missing normalized risk score.")
    if set(results.model) != {"typesafe/jev-1.13-20260917"} or set(results.provider) != {"TypeSafe"}:
        raise ValueError("Saved response model/provider differs from the completed Jev run.")
    if config["requested_model"] != MODEL or config["api_calls_this_execution"] != 552:
        raise ValueError("Unexpected 05b model or completed-call count.")
    if stats.get("source") != "03 train groups only" or len(features) != 20:
        raise ValueError("The expected 20-feature train-only Robust scaler is unavailable.")
    if request_audit["same_robust_features_across_prompts"] is not True:
        raise ValueError("Saved request audit does not confirm paired feature identity.")
    if questions["v4"]["type"] != questions["v5"]["type"] or questions["v4"]["criteria"] != questions["v5"]["criteria"]:
        raise ValueError("v4/v5 question type and criteria differ.")
    if selection["v5_passes_development"] is not False or config["final_exploratory_recommendation"] != "v4_baseline":
        raise ValueError("The saved 05b v4 selection changed.")
    if len(old) != 45 or len(old_result) != 45:
        raise ValueError("Expected the saved 45-row 05a Robust cohort.")
    if sample["target"].isna().any() or set(sample.target) != set(targets):
        raise ValueError("Saved targets differ from the 05a/03 target definition.")
    fixed = pd.read_csv(_required(root / "artifacts" / "random_forest"
                                  / "ttf_baseline_group_split.csv"),
                        usecols=["group_id", "split"])
    validation_groups = set(fixed.loc[fixed.split == "validation", "group_id"].astype(str))
    if not set(sample.group_id.astype(str)).issubset(validation_groups):
        raise ValueError("A selected 05b group is outside the fixed 03 validation split.")
    saved_v4 = json.loads(_required(root / "artifacts" / "jev_tut" / "prompt_v4.txt")
                          .read_text(encoding="utf-8").split("question=\n", 1)[1])
    if questions["v4"] != saved_v4:
        raise ValueError("Saved 05b v4 question differs from the exact 05a v4 question.")
    joined = results.merge(sample[["sample_id", "subset", "target", "region", "group_id", *PHYSICAL]],
                           on="sample_id", validate="many_to_one", suffixes=("_response", ""))
    if len(joined) != 540 or not (joined.subset_response == joined.subset).all():
        raise ValueError("Primary response/sample pairing or subset labels are inconsistent.")
    for subset, prompt, sep, auc, ap, order in [
        ("development", "v4", .0337, .6017, .5922, 1),
        ("development", "v5", .0264, .5838, .5802, 1),
        ("holdout", "v4", .0422, .6217, .6716, 1),
        ("holdout", "v5", .0308, .6050, .6810, 1),
    ]:
        row = macro.loc[(macro.subset == subset) & (macro.prompt_id == prompt)].iloc[0]
        if not (abs(row.separation - sep) < 5e-5 and abs(row.ROC_AUC - auc) < 5e-5
                and abs(row.PR_AUC - ap) < 5e-5 and int(row.ordered_targets) == order):
            raise ValueError(f"Saved 05b macro results differ for {subset}/{prompt}.")
    return out, config, selection, questions, stats, features, targets, sample, joined, repeats, old, old_result, macro


def _sampling_audit(sample: pd.DataFrame, old: pd.DataFrame):
    for name, frame in (("05a", old), ("05b", sample)):
        ttf = pd.to_numeric(frame.ttf_seconds, errors="coerce")
        if ttf.isna().any() or not np.isfinite(ttf).all():
            raise ValueError(f"{name} has invalid TTF values.")
        expected = np.select([ttf <= 5000, ttf <= 50000],
                             ["near_failure", "boundary"], default="normal")
        if not np.array_equal(expected, frame.region.to_numpy()):
            raise ValueError(f"{name} TTF-band labels do not match saved 05a boundaries.")
    reconstructed = sample.source_file.astype(str) + "::" + sample.sequence_index.astype(str)
    if not (reconstructed == sample.group_id.astype(str)).all():
        raise ValueError("Saved 05b group IDs differ from source_file::sequence_index.")
    groups_dev = set(sample.loc[sample.subset == "development", "group_id"])
    groups_hold = set(sample.loc[sample.subset == "holdout", "group_id"])
    group_overlap = groups_dev & groups_hold
    if group_overlap:
        raise ValueError("Development and holdout share source_file::sequence_index groups.")
    old_target = set(zip(old.target, old.source_file, old.sequence_index, old.window_index))
    new_target = set(zip(sample.target, sample.source_file, sample.sequence_index, sample.window_index))
    physical = _physical_keys(sample)
    old_physical = _physical_keys(old)
    rows = []
    for (subset, target, region), d in sample.groupby(["subset", "target", "region"], sort=True):
        counts = d.groupby("group_id").size()
        rows.append({"subset": subset, "target": target, "region": region,
                     "samples": len(d), "unique_groups": len(counts),
                     "max_samples_in_one_group": int(counts.max()),
                     "largest_group_share": float(counts.max() / len(d))})
    audit = {
        "development_samples": int((sample.subset == "development").sum()),
        "holdout_samples": int((sample.subset == "holdout").sum()),
        "development_groups": len(groups_dev), "holdout_groups": len(groups_hold),
        "group_overlap": len(group_overlap),
        "same_physical_window_within_05b": len(sample) - len(physical),
        "same_physical_window_between_05a_and_05b": len(physical & old_physical),
        "same_target_window_between_05a_and_05b": len(old_target & new_target),
        "05a_exclusion_key_in_05b_source": "target + source_file + sequence_index + window_index",
        "physical_window_key": list(PHYSICAL),
        "analysis_unit": "100 observations, step 50, adjacent windows overlap",
        "group_unit": "source_file::sequence_index",
    }
    return pd.DataFrame(rows), audit


def _new_feature_rows(root: Path, sample: pd.DataFrame, features: list[str]):
    """Read only selected processed windows; keep feature vectors in memory."""
    raw_rows: dict[tuple[str, int, int], dict[str, float]] = {}
    for source, block in sample.groupby("source_file", sort=True):
        path = _required(root / "data" / "interim" / "phm2018_jev" / "features" / "train"
                         / source.replace(".csv", "_jev_windows.csv"))
        wanted = _physical_keys(block)
        columns = ["source_file", "sequence_index", "window_index", *features]
        for chunk in pd.read_csv(path, usecols=columns, chunksize=25_000):
            pairs = pd.MultiIndex.from_arrays([chunk.sequence_index, chunk.window_index])
            wanted_pairs = {(seq, win) for _, seq, win in wanted}
            selected = chunk.loc[pairs.isin(wanted_pairs)]
            for row in selected.itertuples(index=False):
                key = (str(row.source_file), int(row.sequence_index), int(row.window_index))
                if key in wanted:
                    if key in raw_rows:
                        raise ValueError(f"Processed feature window appears twice: {key}")
                    raw_rows[key] = {f: float(getattr(row, f)) for f in features}
    if set(raw_rows) != _physical_keys(sample):
        raise ValueError("Could not reconstruct every saved 05b processed window from interim features.")
    return raw_rows


def _robust_values(target: str, raw: dict[str, float], features: list[str], stats: dict):
    scaled = {}
    for feature in features:
        spec = stats["targets"][target][feature]
        value = (float(raw[feature]) - float(spec["median_approx"])) / float(spec["safe_iqr"])
        if not np.isfinite(value):
            raise ValueError(f"Non-finite Robust value for {target}/{feature}.")
        scaled[feature] = float(value)
    return scaled


def _feature_and_payload_audit(sample, old, joined, features, stats, questions, raw_rows):
    result_lookup = joined.set_index(["sample_id", "prompt_id"])
    feature_rows = []
    matching_hashes = 0
    for rec in sample.itertuples(index=False):
        raw = raw_rows[(rec.source_file, int(rec.sequence_index), int(rec.window_index))]
        scaled = _robust_values(rec.target, raw, features, stats)
        for prompt in PROMPTS:
            payload = {"model": MODEL, "state": {"features": scaled},
                       "questions": {"equipment_state": questions[prompt]}}
            packed = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                separators=(",", ":"), allow_nan=False)
            expected = hashlib.sha256(packed.encode("utf-8")).hexdigest()
            saved = result_lookup.loc[(rec.sample_id, prompt), "request_hash"]
            if expected != saved:
                raise ValueError(f"Reconstructed request hash mismatch: {rec.sample_id}/{prompt}")
            matching_hashes += 1
        feature_rows.append(("05b", rec.target, rec.region, scaled))
    for rec in old.itertuples(index=False):
        raw = {f: float(getattr(rec, f)) for f in features}
        feature_rows.append(("05a", rec.target, rec.region,
                             _robust_values(rec.target, raw, features, stats)))
    aggregates = []
    for cohort in ("05a", "05b"):
        for target in sorted(set(sample.target)):
            for region in REGIONS:
                vectors = [v for c, t, r, v in feature_rows if c == cohort and t == target and r == region]
                for f in features:
                    z = np.array([v[f] for v in vectors], dtype=float)
                    aggregates.append({"cohort": cohort, "target": target, "region": region,
                                       "feature": f, "n": len(z), "median_robust": float(np.median(z)),
                                       "median_abs_robust": float(np.median(np.abs(z))),
                                       "p90_abs_robust": float(np.quantile(np.abs(z), .9)),
                                       "share_abs_above_2": float(np.mean(np.abs(z) > 2))})
    return pd.DataFrame(aggregates), matching_hashes


def _pairs(joined: pd.DataFrame) -> pd.DataFrame:
    keys = ["sample_id", "subset", "target", "region", "group_id", *PHYSICAL]
    pair = joined.pivot(index=keys, columns="prompt_id", values="risk_score").reset_index()
    pair.columns.name = None
    if len(pair) != 270 or pair[["v4", "v5"]].isna().any().any():
        raise ValueError("The 270 same-sample v4/v5 pairs are incomplete.")
    pair["v5_minus_v4"] = pair.v5 - pair.v4
    return pair


def _risk_diagnostics(joined: pd.DataFrame, pair: pd.DataFrame, old_result: pd.DataFrame):
    risk_rows = []
    for (subset, prompt, target, region), d in joined.groupby(
            ["subset", "prompt_id", "target", "region"], sort=True):
        risk_rows.append({"subset": subset, "prompt_id": prompt, "target": target,
                          "region": region, "n": len(d), "mean": float(d.risk_score.mean()),
                          "median": float(d.risk_score.median()),
                          "p10": float(d.risk_score.quantile(.1)),
                          "p90": float(d.risk_score.quantile(.9)),
                          "std": float(d.risk_score.std(ddof=1))})
    paired_rows = []
    for (subset, target, region), d in pair.groupby(["subset", "target", "region"], sort=True):
        paired_rows.append({"subset": subset, "target": target, "region": region,
                            "n": len(d), "mean_v5_minus_v4": float(d.v5_minus_v4.mean()),
                            "median_v5_minus_v4": float(d.v5_minus_v4.median()),
                            "mean_abs_delta": float(d.v5_minus_v4.abs().mean()),
                            "p10_delta": float(d.v5_minus_v4.quantile(.1)),
                            "p90_delta": float(d.v5_minus_v4.quantile(.9))})
    group_rows = []
    for (subset, prompt, target, region), d in joined.groupby(
            ["subset", "prompt_id", "target", "region"], sort=True):
        group_rows.append({"subset": subset, "prompt_id": prompt, "target": target,
                           "region": region, "row_mean": float(d.risk_score.mean()),
                           "equal_group_mean": float(d.groupby("group_id").risk_score.mean().mean()),
                           "n_groups": int(d.group_id.nunique())})
    group_means = pd.DataFrame(group_rows)
    old = old_result.groupby(["target", "region"]).risk_score.mean().unstack()
    fresh = joined.loc[joined.prompt_id == "v4"].groupby(["target", "region"]).risk_score.mean().unstack()
    historical = pd.DataFrame({"target": sorted(old.index),
                               "old_05a_separation": (old["near_failure"] - old["normal"]).reindex(sorted(old.index)).values,
                               "fresh_05b_v4_separation": (fresh["near_failure"] - fresh["normal"]).reindex(sorted(old.index)).values})
    historical["unpaired_delta"] = historical.fresh_05b_v4_separation - historical.old_05a_separation
    return pd.DataFrame(risk_rows), pd.DataFrame(paired_rows), group_means, historical


def _macro_separation(frame: pd.DataFrame, prompt: str) -> float:
    d = frame.loc[:, ["target", "region", prompt]]
    means = d.groupby(["target", "region"])[prompt].mean().unstack()
    if means["normal"].isna().any() or means["near_failure"].isna().any():
        return float("nan")
    return float((means["near_failure"] - means["normal"]).mean())


def _group_influence(pair: pd.DataFrame):
    rows = []
    for subset, d in pair.groupby("subset"):
        baseline = {p: _macro_separation(d, p) for p in PROMPTS}
        for group in sorted(d.group_id.unique()):
            remainder = d.loc[d.group_id != group]
            if any(remainder.groupby(["target", "region"]).size().reindex(
                    pd.MultiIndex.from_product([sorted(d.target.unique()), REGIONS]), fill_value=0) == 0):
                continue
            result = {p: _macro_separation(remainder, p) for p in PROMPTS}
            rows.append({"subset": subset, "group_id": group, "rows_removed": int((d.group_id == group).sum()),
                         "v4_separation_change": result["v4"] - baseline["v4"],
                         "v5_separation_change": result["v5"] - baseline["v5"],
                         "paired_delta_change": (result["v5"] - result["v4"])
                                                - (baseline["v5"] - baseline["v4"])})
    return pd.DataFrame(rows)


def _paired_point(frame: pd.DataFrame, weights: np.ndarray | None = None):
    if weights is None:
        weights = np.ones(len(frame), dtype=float)
    out = {}
    for metric in ("separation", "ROC_AUC", "PR_AUC"):
        deltas = []
        for target, d in frame.groupby("target", sort=True):
            w = weights[d.index.to_numpy()]
            regions = d.region.to_numpy()
            def weighted_mean(p, region):
                mask = (regions == region) & (w > 0)
                if not mask.any():
                    raise ValueError("A bootstrap draw omitted a target/region cell.")
                return float(np.average(d.loc[mask, p].to_numpy(), weights=w[mask]))
            if metric == "separation":
                a = [weighted_mean(p, "near_failure") - weighted_mean(p, "normal") for p in PROMPTS]
            else:
                binary = np.isin(regions, ["normal", "near_failure"]) & (w > 0)
                labels = (regions[binary] == "near_failure").astype(int)
                if np.unique(labels).size != 2:
                    raise ValueError("A bootstrap draw omitted a binary comparison class.")
                fn = roc_auc_score if metric == "ROC_AUC" else average_precision_score
                a = [float(fn(labels, d.loc[binary, p].to_numpy(), sample_weight=w[binary]))
                     for p in PROMPTS]
            deltas.append(a[1] - a[0])
        out[metric] = float(np.mean(deltas))
    return out


def _paired_group_bootstrap(pair: pd.DataFrame, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for subset, d0 in pair.groupby("subset", sort=True):
        d = d0.reset_index(drop=True)
        groups = np.array(sorted(d.group_id.unique()))
        group_index = pd.Categorical(d.group_id, categories=groups).codes
        observed = _paired_point(d)
        estimates = {k: [] for k in observed}
        invalid = 0
        for _ in range(n_boot):
            multiplicity = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))
            w = multiplicity[group_index].astype(float)
            try:
                value = _paired_point(d, w)
            except ValueError:
                invalid += 1
                continue
            for metric in estimates:
                estimates[metric].append(value[metric])
        for metric, values in estimates.items():
            rows.append({"subset": subset, "metric": metric, "observed_v5_minus_v4": observed[metric],
                         "ci_2p5": float(np.quantile(values, .025)) if values else np.nan,
                         "ci_97p5": float(np.quantile(values, .975)) if values else np.nan,
                         "valid_replicates": len(values), "invalid_replicates": invalid,
                         "bootstrap_unit": "source_file::sequence_index"})
    return pd.DataFrame(rows)


def _threshold_candidates(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(scores, dtype=float))
    if len(unique) < 3 or np.any(~np.isfinite(unique)) or unique[0] < 0 or unique[-1] > 1:
        raise ValueError("At least three distinct, finite development scores in [0,1] are required.")
    return (unique[:-1] + unique[1:]) / 2


def _choose_thresholds(frame: pd.DataFrame, weights: np.ndarray | None = None):
    """Maximize the equal-target, equal-TTF-band mean recall; ties choose lowest pair."""
    scores = frame.v4.to_numpy(dtype=float)
    if weights is None:
        weights = np.ones(len(frame), dtype=float)
    candidates = _threshold_candidates(scores[weights > 0])
    below = {}
    for (target, region), d in frame.groupby(["target", "region"], sort=True):
        indices = d.index.to_numpy()
        s, w = scores[indices], weights[indices]
        order = np.argsort(s)
        s, w = s[order], w[order]
        total = float(w.sum())
        if total <= 0:
            raise ValueError("A threshold draw omitted a target/region cell.")
        below[(target, region)] = np.r_[0.0, np.cumsum(w)][np.searchsorted(s, candidates, side="left")] / total
    targets = sorted(frame.target.unique())
    low = sum((below[(t, "normal")] - below[(t, "boundary")] for t in targets),
              np.zeros(len(candidates)))
    high = sum((below[(t, "boundary")] - below[(t, "near_failure")] for t in targets),
               np.zeros(len(candidates)))
    objective = (low[:, None] + high[None, :] + len(targets)) / (3 * len(targets))
    objective[np.tril_indices_from(objective)] = -np.inf
    i, j = np.unravel_index(int(np.argmax(objective)), objective.shape)
    if not np.isfinite(objective[i, j]):
        raise ValueError("No ordered pair of threshold midpoints gives three score bands.")
    return float(candidates[i]), float(candidates[j]), float(objective[i, j]), len(candidates)


def _threshold_metrics(frame: pd.DataFrame, t1: float, t2: float, subset: str,
                       predicted_col: str | None = None):
    rows = []
    confusions = []
    for target, d in frame.groupby("target", sort=True):
        truth = pd.Categorical(d.region, categories=REGIONS, ordered=True).codes
        predicted = (d[predicted_col].to_numpy(dtype=int) if predicted_col else
                     np.searchsorted([t1, t2], d.v4.to_numpy(), side="right"))
        cm = confusion_matrix(truth, predicted, labels=[0, 1, 2])
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predicted, labels=[0, 1, 2], zero_division=0)
        rows.append({"subset": subset, "target": target, "n": len(d),
                     "balanced_accuracy": float(np.mean(recall)), "macro_F1": float(np.mean(f1)),
                     "normal_high_rate": float(cm[0, 2] / cm[0].sum()),
                     "near_failure_low_rate": float(cm[2, 0] / cm[2].sum()),
                     "middle_prediction_rate": float(np.mean(predicted == 1))})
        for true_i, true_name in enumerate(REGIONS):
            for pred_i, pred_name in enumerate(REGIONS):
                confusions.append({"subset": subset, "target": target,
                                   "truth_region": true_name, "predicted_band": pred_name,
                                   "count": int(cm[true_i, pred_i]),
                                   "precision_predicted": float(precision[pred_i]),
                                   "recall_true": float(recall[true_i])})
    macro = {"subset": subset, "target": "macro", "n": len(frame)}
    for name in ("balanced_accuracy", "macro_F1", "normal_high_rate",
                 "near_failure_low_rate", "middle_prediction_rate"):
        macro[name] = float(np.mean([r[name] for r in rows]))
    rows.append(macro)
    return pd.DataFrame(rows), pd.DataFrame(confusions)


def _constant_baselines(frame: pd.DataFrame):
    rows = []
    for region in REGIONS:
        cls = REGIONS.index(region)
        target_rows = []
        for _, d in frame.groupby("target"):
            truth = pd.Categorical(d.region, categories=REGIONS, ordered=True).codes
            predicted = np.full(len(d), cls)
            _, recall, f1, _ = precision_recall_fscore_support(
                truth, predicted, labels=[0, 1, 2], zero_division=0)
            target_rows.append((float(np.mean(recall)), float(np.mean(f1))))
        rows.append({"constant_prediction": region,
                     "macro_balanced_accuracy": float(np.mean([v[0] for v in target_rows])),
                     "macro_F1": float(np.mean([v[1] for v in target_rows]))})
    return pd.DataFrame(rows)


def _group_cv(dev: pd.DataFrame):
    dev = dev.reset_index(drop=True)
    required_cells = pd.MultiIndex.from_product([sorted(dev.target.unique()), REGIONS])
    candidate_splits = None
    chosen_k = None
    for folds in (5, 4, 3, 2):
        if dev.group_id.nunique() < folds:
            continue
        split = list(GroupKFold(n_splits=folds).split(dev, groups=dev.group_id))
        if all((dev.iloc[tr].groupby(["target", "region"]).size()
                .reindex(required_cells, fill_value=0) > 0).all() and
               (dev.iloc[te].groupby(["target", "region"]).size()
                .reindex(required_cells, fill_value=0) > 0).all()
               for tr, te in split):
            candidate_splits, chosen_k = split, folds
            break
    if candidate_splits is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "status": "infeasible", "reason": "No 2-5 fold GroupKFold has all nine target/TTF cells in every train and test fold."}
    oof = dev.copy()
    oof["predicted_class"] = -1
    fold_rows, fold_metrics = [], []
    for fold, (tr, te) in enumerate(candidate_splits, 1):
        train = dev.iloc[tr].reset_index(drop=True)
        test = dev.iloc[te]
        t1, t2, train_objective, count = _choose_thresholds(train)
        oof.loc[te, "predicted_class"] = np.searchsorted([t1, t2], test.v4.to_numpy(), side="right")
        fm, _ = _threshold_metrics(test, t1, t2, f"cv_fold_{fold}")
        fold_metrics.append(fm)
        fold_rows.append({"fold": fold, "train_rows": len(train), "test_rows": len(test),
                          "train_groups": int(train.group_id.nunique()),
                          "test_groups": int(test.group_id.nunique()),
                          "t1": t1, "t2": t2, "train_objective": train_objective,
                          "candidate_threshold_count": count,
                          "group_overlap": len(set(train.group_id) & set(test.group_id))})
    if (oof.predicted_class < 0).any():
        raise AssertionError("Some development rows were omitted from the out-of-fold predictions.")
    oof_metrics, oof_confusion = _threshold_metrics(oof, 0.0, 1.0, "development_group_cv_oof",
                                                    predicted_col="predicted_class")
    return pd.DataFrame(fold_rows), pd.concat(fold_metrics, ignore_index=True), oof_metrics, {
        "status": "feasible", "fold_count": chosen_k,
        "fold_assignment_rule": "largest k in 5,4,3,2 with every target/region present in both sides",
        "group_overlap_in_any_fold": False,
        "out_of_fold_confusion": oof_confusion.to_dict(orient="records")}


def _threshold_bootstrap(dev: pd.DataFrame, holdout: pd.DataFrame,
                         t1: float, t2: float, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for subset, frame0 in (("development_reselection", dev), ("holdout_locked_thresholds", holdout)):
        frame = frame0.reset_index(drop=True)
        groups = np.array(sorted(frame.group_id.unique()))
        group_index = pd.Categorical(frame.group_id, categories=groups).codes
        valid, invalid = [], 0
        for _ in range(n_boot):
            multiplicity = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))
            weights = multiplicity[group_index].astype(float)
            try:
                if subset == "development_reselection":
                    a, b, objective, _ = _choose_thresholds(frame, weights)
                    valid.append((a, b, objective))
                else:
                    pred = np.searchsorted([t1, t2], frame.v4.to_numpy(), side="right")
                    acc = []
                    for target in sorted(frame.target.unique()):
                        for label, region in enumerate(REGIONS):
                            m = (frame.target.to_numpy() == target) & (frame.region.to_numpy() == region)
                            if weights[m].sum() == 0:
                                raise ValueError("A bootstrap draw omitted a target/region cell.")
                            acc.append(float(np.average(pred[m] == label, weights=weights[m])))
                    valid.append((float(np.mean(acc)),))
            except ValueError:
                invalid += 1
        labels = ("t1", "t2", "inbag_macro_balanced_accuracy") if subset == "development_reselection" else ("holdout_macro_balanced_accuracy",)
        arr = np.asarray(valid, dtype=float)
        for i, label in enumerate(labels):
            rows.append({"bootstrap_analysis": subset, "metric": label,
                         "ci_2p5": float(np.quantile(arr[:, i], .025)) if len(arr) else np.nan,
                         "median": float(np.median(arr[:, i])) if len(arr) else np.nan,
                         "ci_97p5": float(np.quantile(arr[:, i], .975)) if len(arr) else np.nan,
                         "valid_replicates": len(valid), "invalid_replicates": invalid,
                         "bootstrap_unit": "source_file::sequence_index"})
    return pd.DataFrame(rows)


def _repeat_crossings(repeats: pd.DataFrame, t1: float, t2: float):
    d = repeats.loc[repeats.prompt_id == "v4"].copy()
    d["primary_band"] = np.searchsorted([t1, t2], d.baseline_risk.to_numpy(), side="right")
    d["repeat_band"] = np.searchsorted([t1, t2], d.repeat_risk.to_numpy(), side="right")
    d["changed_band"] = d.primary_band != d.repeat_band
    return d


def _make_plots(diagnostics: Path, joined: pd.DataFrame, pair: pd.DataFrame,
                features: pd.DataFrame, t1: float, t2: float):
    targets = sorted(joined.target.unique())
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True, constrained_layout=True)
    for row, subset in enumerate(("development", "holdout")):
        for col, target in enumerate(targets):
            ax = axes[row, col]
            d = joined.loc[(joined.subset == subset) & (joined.target == target)]
            data, positions = [], []
            for i, region in enumerate(REGIONS):
                for j, prompt in enumerate(PROMPTS):
                    data.append(d.loc[(d.region == region) & (d.prompt_id == prompt), "risk_score"].to_numpy())
                    positions.append(i * 3 + j + 1)
            parts = ax.boxplot(data, positions=positions, widths=.7, patch_artist=True,
                               showfliers=True)
            for k, box in enumerate(parts["boxes"]):
                box.set_facecolor("#4c78a8" if k % 2 == 0 else "#f58518")
                box.set_alpha(.7)
            ax.set_xticks([1.5, 4.5, 7.5], ["normal", "boundary", "near"], rotation=20)
            ax.axhline(t1, color="#7f7f7f", linestyle="--", linewidth=1)
            ax.axhline(t2, color="#333333", linestyle=":", linewidth=1)
            ax.set_title(f"{subset}: {target.removeprefix('ttf_flowcool_').removesuffix('_seconds')}")
            if col == 0:
                ax.set_ylabel("Normalized Jev risk (v4 blue, v5 orange)")
    fig.savefig(diagnostics / "risk_score_distributions.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True, constrained_layout=True)
    for ax, (subset, d) in zip(axes, pair.groupby("subset", sort=True)):
        datasets = [d.loc[d.region == r, "v5_minus_v4"].to_numpy() for r in REGIONS]
        ax.boxplot(datasets, tick_labels=REGIONS)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"Same-window v5 minus v4: {subset}")
        ax.tick_params(axis="x", rotation=15)
    axes[0].set_ylabel("Risk score difference")
    fig.savefig(diagnostics / "paired_prompt_deltas.png", dpi=150)
    plt.close(fig)

    summary = features.groupby(["cohort", "feature"]).median_abs_robust.mean().unstack("cohort")
    summary["05b_minus_05a"] = summary["05b"] - summary["05a"]
    order = summary["05b_minus_05a"].abs().sort_values(ascending=False).head(12).index
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    y = np.arange(len(order))
    ax.barh(y - .18, summary.loc[order, "05a"], height=.35, label="05a (5 per target/band)")
    ax.barh(y + .18, summary.loc[order, "05b"], height=.35, label="05b (30 per target/band)")
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlabel("Mean of target/band median |train-only Robust value|")
    ax.legend()
    fig.savefig(diagnostics / "feature_magnitude_cohorts.png", dpi=150)
    plt.close(fig)


def run_diagnostics(root: Path, bootstrap_replicates: int = 1000) -> dict:
    """Run the saved-response audit and exploratory threshold analysis offline."""
    root = Path(root).resolve()
    (out, config, selection, questions, stats, features, targets, sample, joined,
     repeats, old, old_result, macro) = _check_inputs(root)
    diagnostics = out / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    sampling_table, sampling = _sampling_audit(sample, old)
    raw_rows = _new_feature_rows(root, sample, features)
    feature_table, hash_count = _feature_and_payload_audit(
        sample, old, joined, features, stats, questions, raw_rows)
    del raw_rows  # Only aggregate feature summaries are written.
    pair = _pairs(joined)
    risk_table, delta_table, group_means, historical = _risk_diagnostics(joined, pair, old_result)
    group_macro_rows = []
    for (subset, prompt), d in group_means.groupby(["subset", "prompt_id"]):
        row_means = d.pivot(index="target", columns="region", values="row_mean")
        equal_means = d.pivot(index="target", columns="region", values="equal_group_mean")
        group_macro_rows.append({"subset": subset, "prompt_id": prompt,
                                 "row_weighted_macro_separation": float((row_means.near_failure - row_means.normal).mean()),
                                 "equal_group_macro_separation": float((equal_means.near_failure - equal_means.normal).mean())})
    group_macro = pd.DataFrame(group_macro_rows)
    influence = _group_influence(pair)
    paired_ci = _paired_group_bootstrap(pair, bootstrap_replicates, seed=42)
    dev = pair.loc[pair.subset == "development"].reset_index(drop=True)
    hold = pair.loc[pair.subset == "holdout"].reset_index(drop=True)
    cv_folds, cv_fold_metrics, cv_oof_metrics, cv_status = _group_cv(dev)
    t1, t2, dev_objective, n_candidates = _choose_thresholds(dev)
    dev_metrics, dev_confusion = _threshold_metrics(dev, t1, t2, "development_apparent")
    hold_metrics, hold_confusion = _threshold_metrics(hold, t1, t2, "holdout_reused")
    threshold_ci = _threshold_bootstrap(dev, hold, t1, t2, bootstrap_replicates, seed=43)
    constant = _constant_baselines(dev)
    repeat_crossings = _repeat_crossings(repeats, t1, t2)
    predictions = pair[["sample_id", "subset", "target", "region", "group_id", "v4"]].copy()
    predictions["predicted_band"] = np.asarray(REGIONS)[
        np.searchsorted([t1, t2], predictions.v4.to_numpy(), side="right")]
    confusion = pd.concat([dev_confusion, hold_confusion], ignore_index=True)
    if cv_status["status"] == "feasible":
        confusion = pd.concat([confusion, pd.DataFrame(cv_status["out_of_fold_confusion"])],
                              ignore_index=True)
    class_rows = []
    for (subset, target, region), d in confusion.groupby(["subset", "target", "truth_region"]):
        diagonal = d.loc[d.predicted_band == region]
        tp = int(diagonal["count"].iloc[0])
        class_rows.append({"subset": subset, "target": target, "region": region,
                           "support": int(d["count"].sum()),
                           "precision": float(diagonal.precision_predicted.iloc[0]),
                           "recall": float(diagonal.recall_true.iloc[0]),
                           "true_positive_count": tp})

    frames = {
        "sample_group_audit.csv": sampling_table,
        "feature_robust_distribution_summary.csv": feature_table,
        "risk_distribution_summary.csv": risk_table,
        "paired_prompt_delta_summary.csv": delta_table,
        "row_vs_equal_group_means.csv": group_means,
        "row_vs_equal_group_macro_separation.csv": group_macro,
        "05a_vs_05b_unpaired_separation.csv": historical,
        "leave_one_group_out_influence.csv": influence,
        "paired_group_bootstrap_observed_and_ci.csv": paired_ci,
        "threshold_cv_folds.csv": cv_folds,
        "threshold_cv_fold_metrics.csv": cv_fold_metrics,
        "threshold_metrics.csv": pd.concat([dev_metrics, hold_metrics, cv_oof_metrics], ignore_index=True),
        "threshold_confusion.csv": pd.concat([dev_confusion, hold_confusion], ignore_index=True),
        "threshold_class_metrics.csv": pd.DataFrame(class_rows),
        "threshold_sample_predictions.csv": predictions,
        "threshold_constant_baselines.csv": constant,
        "threshold_group_bootstrap_ci.csv": threshold_ci,
        "repeat_threshold_crossings.csv": repeat_crossings,
    }
    for name, frame in frames.items():
        frame.to_csv(diagnostics / name, index=False)
    _make_plots(diagnostics, joined, pair, feature_table, t1, t2)
    old_macro = float(historical.old_05a_separation.mean())
    fresh_macro = float(historical.fresh_05b_v4_separation.mean())
    old_near_ttf = float(old.loc[old.region == "near_failure", "ttf_seconds"].median())
    new_near_ttf = float(sample.loc[sample.region == "near_failure", "ttf_seconds"].median())
    old_groups = int(old.loc[:, ["source_file", "sequence_index"]].drop_duplicates().shape[0])
    v4_dev_group = group_macro.loc[(group_macro.subset == "development") &
                                   (group_macro.prompt_id == "v4")].iloc[0]
    ci_t1 = threshold_ci.loc[threshold_ci.metric == "t1"].iloc[0]
    ci_t2 = threshold_ci.loc[threshold_ci.metric == "t2"].iloc[0]
    run = {
        "analysis_mode": "offline_saved_responses_only",
        "additional_api_calls": 0,
        "original_05b_api_calls": int(config["api_calls_this_execution"]),
        "original_05b_cache_hits": int(config["cache_hits_this_execution"]),
        "original_05b_parse_errors": int(config["parse_errors_this_execution"]),
        "original_05b_input_tokens": int(config["input_tokens_reported_this_execution"]),
        "original_05b_output_tokens": int(config["output_tokens_reported_this_execution"]),
        "original_05b_reported_cost_usd": float(config["reported_cost_usd_this_execution"]),
        "original_05b_models": config["returned_model_counts_this_execution"],
        "original_05b_providers": config["provider_counts_this_execution"],
        "sampling_audit": sampling,
        "reconstructed_primary_request_hashes_matching": hash_count,
        "question_only_change_verified": hash_count == 540,
        "v5_development_selected": bool(selection["v5_passes_development"]),
        "v5_development_failure": selection["criteria"],
        "old_05a_macro_separation": old_macro,
        "fresh_05b_v4_macro_separation": fresh_macro,
        "unpaired_separation_change": fresh_macro - old_macro,
        "old_05a_unique_groups": old_groups,
        "old_05a_near_ttf_median_seconds": old_near_ttf,
        "new_05b_near_ttf_median_seconds": new_near_ttf,
        "thresholds": {"t1": t1, "t2": t2, "development_objective": dev_objective,
                       "unique_midpoint_candidates": n_candidates,
                       "selection_data": "development 180 rows only",
                       "selection_metric": "mean of each target's normal/boundary/near_failure recall",
                       "candidate_rule": "midpoints between sorted distinct development v4 scores",
                       "tie_rule": "lexicographically lowest (t1,t2) among maximum objectives",
                       "missing_rule": "stop if any selected response/score/TTF band is missing",
                       "actual_current_state_labels_available": False,
                       "operational_adoption": "deferred"},
        "group_cv": cv_status,
        "bootstrap_replicates_requested": bootstrap_replicates,
        "repeat_v4_band_changes": int(repeat_crossings.changed_band.sum()),
        "repeat_v4_count": len(repeat_crossings),
        "limitations": [
            "TTF bands are proxy labels for future event time, not observed present equipment condition.",
            "The 05b holdout was already viewed for prompt selection; threshold use here is secondary analysis.",
            "Balanced TTF sampling does not represent operating prevalence or predictive value.",
            "Adjacent 100-observation windows overlap by 50 observations within a run.",
            "The historical 05a and 05b cohorts are disjoint and their difference is unpaired.",
            "Jev risk scores and confidence are not calibrated physical failure probabilities.",
        ],
    }
    _write_json(diagnostics / "analysis_config_and_summary.json", run)
    _write_json(diagnostics / "sampling_audit.json", sampling)
    if cv_status["status"] == "feasible":
        _write_json(diagnostics / "group_cv_status.json", cv_status)
    summary_lines = [
        "# 05b offline diagnostics and exploratory three-band thresholds", "",
        f"- Original run: {config['api_calls_this_execution']} API calls, {config['cache_hits_this_execution']} cache hits, "
        f"{config['input_tokens_reported_this_execution']} input / {config['output_tokens_reported_this_execution']} output tokens, "
        f"USD {config['reported_cost_usd_this_execution']:.9f}. This analysis made 0 API calls.",
        f"- Saved requests reconstructed and hash-matched: {hash_count}/540; development/holdout group overlap: {sampling['group_overlap']}; physical-window duplicates: {sampling['same_physical_window_within_05b']}.",
        f"- 05a Robust macro gap {old_macro:.4f} (45 rows, {old_groups} groups) vs 05b v4 {fresh_macro:.4f} (270 rows, {sampling['development_groups'] + sampling['holdout_groups']} groups). The cohorts do not share physical windows. This difference is unpaired.",
        f"- The cohorts also differ in TTF distribution (near-failure median {old_near_ttf:.0f}s vs {new_near_ttf:.0f}s) and in several transformed-feature distributions; these facts do not identify the cause of the gap.",
        f"- V5 failed the locked development rule: v5-v4 macro gap change {float(paired_ci.loc[(paired_ci.subset == 'development') & (paired_ci.metric == 'separation'), 'observed_v5_minus_v4'].iloc[0]):+.4f}. Retain v4 + train-only Robust 1x as the saved prompt choice.",
        f"- Row-weighted vs equal-group v4 development macro gap: {v4_dev_group.row_weighted_macro_separation:.4f} vs {v4_dev_group.equal_group_macro_separation:.4f}. Holdout leak near-failure has only {int(sampling_table.loc[(sampling_table.subset == 'holdout') & (sampling_table.target == 'ttf_flowcool_leak_seconds') & (sampling_table.region == 'near_failure'), 'unique_groups'].iloc[0])} independent groups.",
        f"- Exploratory v4 thresholds: t1={t1:.4f} (group-bootstrap 95% interval {ci_t1.ci_2p5:.4f}–{ci_t1.ci_97p5:.4f}), t2={t2:.4f} ({ci_t2.ci_2p5:.4f}–{ci_t2.ci_97p5:.4f}). These intervals include large shifts in both cutoffs.",
        f"- Group CV: {cv_status['status']}" + (f", {cv_status['fold_count']} folds" if cv_status['status'] == 'feasible' else ""),
        f"- Macro balanced accuracy: development apparent {float(dev_metrics.loc[dev_metrics.target == 'macro', 'balanced_accuracy'].iloc[0]):.4f}; "
        + (f"development group CV {float(cv_oof_metrics.loc[cv_oof_metrics.target == 'macro', 'balanced_accuracy'].iloc[0]):.4f}; " if len(cv_oof_metrics) else "")
        + f"reused holdout {float(hold_metrics.loc[hold_metrics.target == 'macro', 'balanced_accuracy'].iloc[0]):.4f}; constant baseline {float(constant.macro_balanced_accuracy.iloc[0]):.4f}.",
        f"- Reused holdout: normal wrongly assigned high risk {float(hold_metrics.loc[hold_metrics.target == 'macro', 'normal_high_rate'].iloc[0]):.1%}; near-failure wrongly assigned low risk {float(hold_metrics.loc[hold_metrics.target == 'macro', 'near_failure_low_rate'].iloc[0]):.1%}.",
        f"- V4 repeat band changes at these thresholds: {int(repeat_crossings.changed_band.sum())}/{len(repeat_crossings)}; this repeat sample is too small to establish stability.",
        "- TTF bands are surrogate labels. The holdout was already viewed; no untouched external groups or observed current-state fault labels establish an operational three-stage gate.",
        "- Decision: thresholds are exploratory candidates; operational adoption is deferred. Prioritize unused groups and measured current-state labels in the next study.",
    ]
    (diagnostics / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return run
