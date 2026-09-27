"""Offline, group-aware analysis for the exploratory Jev Noul + v4 comparison.

TTF bands are surrogate labels. Review is a separate prediction outcome, counted
as incorrect in three-class recall/F1 and reported as its own coverage measure.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

REGIONS = ("normal", "boundary", "near_failure")
PREDICTIONS = (*REGIONS, "review")
N_LOW = 0.4
N_HIGH = 0.6
TIE_TOL = 1e-12
SEED = 42


def _truth(frame):
    return pd.Categorical(frame.region, categories=REGIONS).codes.astype(int)


def _target(frame):
    return pd.Categorical(frame.target, categories=sorted(frame.target.unique())).codes.astype(int)


def _score_values(frame):
    return pd.to_numeric(frame.risk_score, errors="coerce").to_numpy(dtype=float)


def _noul_values(frame):
    return pd.to_numeric(frame.noul_value, errors="coerce").to_numpy(dtype=float)


def predict_baseline(score, t1, t2):
    score = np.asarray(score, dtype=float)
    pred = np.full(len(score), 3, dtype=int)
    ok = np.isfinite(score)
    pred[ok & (score <= t1)] = 0
    pred[ok & (score > t1) & (score <= t2)] = 1
    pred[ok & (score > t2)] = 2
    return pred


def predict_candidate(score, noul, t1, t2):
    score, noul = np.asarray(score, dtype=float), np.asarray(noul, dtype=float)
    pred = np.full(len(score), 3, dtype=int)
    ok = np.isfinite(score) & np.isfinite(noul)
    pred[ok & (noul < N_LOW)] = 0
    active = ok & (noul > N_HIGH)
    bins = predict_baseline(score, t1, t2)
    pred[active] = bins[active]
    return pred


def _matrix(truth, pred, weights):
    return np.bincount(truth * 4 + pred, weights=weights, minlength=12).reshape(3, 4)


def _from_matrix(matrix):
    support = matrix.sum(axis=1)
    tp = matrix[np.arange(3), np.arange(3)]
    predicted = matrix[:, :3].sum(axis=0)
    recall = np.divide(tp, support, out=np.full(3, np.nan), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(3), where=predicted > 0)
    f1 = np.divide(2 * precision * np.nan_to_num(recall), precision + np.nan_to_num(recall),
                   out=np.zeros(3), where=(precision + np.nan_to_num(recall)) > 0)
    automatic = matrix[:, :3].sum()
    correct = tp.sum()
    values = {
        "balanced_accuracy": float(np.mean(recall)) if np.isfinite(recall).all() else float("nan"),
        "macro_F1": float(np.mean(f1)) if np.isfinite(recall).all() else float("nan"),
        "normal_high_rate": float(matrix[0, 2] / support[0]) if support[0] > 0 else float("nan"),
        "near_failure_low_rate": float(matrix[2, 0] / support[2]) if support[2] > 0 else float("nan"),
        "review_fraction": float(matrix[:, 3].sum() / matrix.sum()) if matrix.sum() > 0 else float("nan"),
        "automatic_fraction": float(automatic / matrix.sum()) if matrix.sum() > 0 else float("nan"),
        "automatic_error_rate": float((automatic - correct) / automatic) if automatic > 0 else float("nan"),
    }
    return values, precision, recall


def macro_metrics(frame, pred_column, weights=None):
    if weights is None:
        weights = np.ones(len(frame), dtype=float)
    truth, pred = _truth(frame), frame[pred_column].to_numpy(dtype=int)
    targets = sorted(frame.target.unique())
    rows = []
    for target in targets:
        mask = (frame.target == target).to_numpy()
        matrix = _matrix(truth[mask], pred[mask], weights[mask])
        values, _, _ = _from_matrix(matrix)
        near_mask = mask & (truth == 2)
        noul = _noul_values(frame)
        gate_miss = float(weights[near_mask & np.isfinite(noul) & (noul < N_LOW)].sum())
        near_total = float(weights[near_mask].sum())
        values.update({"target":target, "n_weighted":float(weights[mask].sum()),
                       "gate_missed_near_failure_count":gate_miss if pred_column == "pred_candidate" else float("nan"),
                       "gate_missed_near_failure_rate":gate_miss / near_total if near_total and pred_column == "pred_candidate" else float("nan")})
        rows.append(values)
    numeric = ["balanced_accuracy", "macro_F1", "normal_high_rate", "near_failure_low_rate",
               "review_fraction", "automatic_fraction", "automatic_error_rate", "gate_missed_near_failure_rate"]
    macro = {key:float(np.mean([row[key] for row in rows])) for key in numeric}
    macro["target"] = "macro"
    macro["n_weighted"] = float(sum(row["n_weighted"] for row in rows))
    macro["gate_missed_near_failure_count"] = (float(sum(row["gate_missed_near_failure_count"] for row in rows))
                                                  if pred_column == "pred_candidate" else float("nan"))
    return rows + [macro]


def _objective(truth, target, pred):
    bas, f1s = [], []
    for code in np.unique(target):
        mask = target == code
        values, _, _ = _from_matrix(_matrix(truth[mask], pred[mask], np.ones(mask.sum())))
        bas.append(values["balanced_accuracy"])
        f1s.append(values["macro_F1"])
    return float(np.mean(bas)), float(np.mean(f1s))


def select_thresholds(frame, method):
    if method not in ("baseline", "candidate"):
        raise ValueError(method)
    score, noul = _score_values(frame), _noul_values(frame)
    valid = np.unique(score[np.isfinite(score)])
    if len(valid) < 2:
        raise ValueError("Threshold search needs at least two distinct Score values.")
    mids = (valid[:-1] + valid[1:]) / 2
    cuts = sorted(set([0.0, 1.0, *mids.tolist()]))
    truth, target = _truth(frame), _target(frame)
    best = None
    tried = 0
    for i, t1 in enumerate(cuts):
        for t2 in cuts[i+1:]:
            tried += 1
            pred = (predict_baseline(score, t1, t2) if method == "baseline"
                    else predict_candidate(score, noul, t1, t2))
            ba, f1 = _objective(truth, target, pred)
            if not (math.isfinite(ba) and math.isfinite(f1)):
                continue
            if (best is None or ba > best[0] + TIE_TOL or
                (abs(ba - best[0]) <= TIE_TOL and f1 > best[1] + TIE_TOL)):
                best = (ba, f1, float(t1), float(t2))
    if best is None:
        raise ValueError("No valid threshold pair.")
    return {"method":method, "t1":best[2], "t2":best[3],
            "training_macro_balanced_accuracy":best[0], "training_macro_F1":best[1],
            "candidate_pairs":tried, "selection_rule":"maximize mean target BA; then mean target macro-F1; then lowest (t1,t2)"}


def assign_predictions(frame, thresholds):
    result = frame.copy()
    score, noul = _score_values(result), _noul_values(result)
    b, c = thresholds["baseline"], thresholds["candidate"]
    result["pred_baseline"] = predict_baseline(score, b["t1"], b["t2"])
    result["pred_candidate"] = predict_candidate(score, noul, c["t1"], c["t2"])
    result["baseline_band"] = [PREDICTIONS[i] for i in result.pred_baseline]
    result["candidate_band"] = [PREDICTIONS[i] for i in result.pred_candidate]
    return result


def three_class_tables(frame, cohort):
    metrics, confusion, classes = [], [], []
    for method, column in (("baseline", "pred_baseline"), ("candidate", "pred_candidate")):
        summary = macro_metrics(frame, column)
        for row in summary:
            metrics.append({"cohort":cohort, "method":method, **row})
        truth, pred = _truth(frame), frame[column].to_numpy(dtype=int)
        for target in sorted(frame.target.unique()):
            mask = (frame.target == target).to_numpy()
            matrix = _matrix(truth[mask], pred[mask], np.ones(mask.sum()))
            _, precision, recall = _from_matrix(matrix)
            for i, region in enumerate(REGIONS):
                for j, predicted in enumerate(PREDICTIONS):
                    confusion.append({"cohort":cohort, "method":method, "target":target,
                                      "truth_region":region, "predicted":predicted, "count":int(matrix[i, j])})
                classes.append({"cohort":cohort, "method":method, "target":target,
                                "class":region, "support":int(matrix[i].sum()),
                                "precision":float(precision[i]), "recall":float(recall[i])})
        for region in REGIONS:
            relevant = [r for r in classes if r["cohort"] == cohort and r["method"] == method and r["class"] == region]
            classes.append({"cohort":cohort, "method":method, "target":"macro", "class":region,
                            "support":sum(r["support"] for r in relevant),
                            "precision":float(np.mean([r["precision"] for r in relevant])),
                            "recall":float(np.mean([r["recall"] for r in relevant]))})
    return pd.DataFrame(metrics), pd.DataFrame(confusion), pd.DataFrame(classes)


def continuous_development(frame):
    rows = []
    for target, block in frame.groupby("target", sort=True):
        binary = block.loc[block.region.isin(["normal", "near_failure"])].copy()
        y = (binary.region == "near_failure").astype(int).to_numpy()
        for method, column in (("v4_score", "risk_score"), ("noul", "noul_value")):
            x = pd.to_numeric(binary[column], errors="coerce").to_numpy(dtype=float)
            row = {"target":target, "method":method, "normal_n":int((y == 0).sum()),
                   "near_failure_n":int((y == 1).sum()),
                   "ROC_AUC_normal_vs_near":float(roc_auc_score(y, x)),
                   "PR_AUC_balanced_sample":float(average_precision_score(y, x)),
                   "mean_normal":float(x[y == 0].mean()), "mean_near_failure":float(x[y == 1].mean())}
            if method == "noul":
                pred = (x >= 0.5).astype(int)
                row["binary_balanced_accuracy_at_0p5"] = float(((pred[y == 0] == 0).mean() + (pred[y == 1] == 1).mean()) / 2)
                row["binary_macro_F1_at_0p5"] = float(f1_score(y, pred, average="macro", zero_division=0))
                row["near_failure_precision_at_0p5"] = float(np.sum((pred == 1) & (y == 1)) / max(np.sum(pred == 1), 1))
                row["near_failure_recall_at_0p5"] = float((pred[y == 1] == 1).mean())
                n_all = _noul_values(block)
                row["review_band_fraction_all_three_TTF_cells"] = float(np.mean(np.isfinite(n_all) & (n_all >= N_LOW) & (n_all <= N_HIGH)))
            rows.append(row)
    table = pd.DataFrame(rows)
    for method in ("v4_score", "noul"):
        part = table.loc[table.method == method]
        numeric = part.select_dtypes(include=["number"]).mean().to_dict()
        rows.append({"target":"macro", "method":method, **numeric})
    return pd.DataFrame(rows)


def group_cv(frame):
    frame = frame.reset_index(drop=True).copy()
    stratum = frame.target.astype(str) + "|" + frame.region.astype(str)
    all_cells = set(stratum)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    splits = list(splitter.split(np.zeros(len(frame)), stratum, frame.group_id))
    for train, test in splits:
        if set(stratum.iloc[train]) != all_cells or set(stratum.iloc[test]) != all_cells:
            raise ValueError("Three-fold grouped CV does not retain every target/TTF cell in train and test.")
        if set(frame.group_id.iloc[train]) & set(frame.group_id.iloc[test]):
            raise ValueError("Grouped CV fold has overlapping groups.")
    rows, fold_rules = [], []
    for number, (train, test) in enumerate(splits, 1):
        training = frame.iloc[train]
        held = frame.iloc[test]
        thresholds = {method:select_thresholds(training, method) for method in ("baseline", "candidate")}
        assigned = assign_predictions(held, thresholds)
        assigned["fold"] = number
        rows.append(assigned)
        for method, chosen in thresholds.items():
            fold_rules.append({"fold":number, "method":method, "train_rows":len(training), "test_rows":len(held),
                               "train_groups":training.group_id.nunique(), "test_groups":held.group_id.nunique(),
                               "group_overlap":0, **chosen})
    oof = pd.concat(rows, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    if len(oof) != len(frame) or oof.sample_id.duplicated().any():
        raise ValueError("Grouped CV does not produce one out-of-fold prediction per row.")
    return oof, pd.DataFrame(fold_rules)


def paired_group_bootstrap(frame, cohort, n_boot=1000):
    groups = np.asarray(sorted(frame.group_id.unique()))
    code = pd.Categorical(frame.group_id, categories=groups).codes
    rng = np.random.default_rng(SEED)
    metric_names = ("balanced_accuracy", "macro_F1", "normal_high_rate", "near_failure_low_rate")
    def estimate(weights):
        b = macro_metrics(frame, "pred_baseline", weights)[-1]
        c = macro_metrics(frame, "pred_candidate", weights)[-1]
        values = {f"candidate_minus_baseline_{key}":c[key]-b[key] for key in metric_names}
        values.update({"candidate_review_fraction":c["review_fraction"],
                       "candidate_automatic_fraction":c["automatic_fraction"],
                       "candidate_automatic_error_rate":c["automatic_error_rate"],
                       "candidate_gate_missed_near_failure_rate":c["gate_missed_near_failure_rate"]})
        return values
    observed = estimate(np.ones(len(frame)))
    draws = {key:[] for key in observed}
    for _ in range(n_boot):
        multiplicity = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))
        values = estimate(multiplicity[code].astype(float))
        for key, value in values.items():
            if math.isfinite(value):
                draws[key].append(value)
    rows = []
    for key, point in observed.items():
        values = np.asarray(draws[key], dtype=float)
        rows.append({"cohort":cohort, "metric":key, "estimate_raw_sample":point,
                     "ci95_low":float(np.quantile(values, .025)) if len(values) else float("nan"),
                     "ci95_high":float(np.quantile(values, .975)) if len(values) else float("nan"),
                     "valid_replicates":len(values), "requested_replicates":n_boot,
                     "bootstrap_unit":"source_file::sequence_index; paired rows and predictions"})
    return pd.DataFrame(rows)


def leave_one_group_out(frame, cohort):
    rows = []
    for group in sorted(frame.group_id.unique()):
        d = frame.loc[frame.group_id != group]
        b, c = macro_metrics(d, "pred_baseline")[-1], macro_metrics(d, "pred_candidate")[-1]
        rows.append({"cohort":cohort, "removed_group":group,
                     "removed_rows":int((frame.group_id == group).sum()),
                     "candidate_minus_baseline_balanced_accuracy":c["balanced_accuracy"]-b["balanced_accuracy"],
                     "candidate_minus_baseline_macro_F1":c["macro_F1"]-b["macro_F1"],
                     "candidate_review_fraction":c["review_fraction"],
                     "candidate_gate_missed_near_failure_rate":c["gate_missed_near_failure_rate"]})
    return pd.DataFrame(rows)


def distribution_summary(frame, cohort):
    rows = []
    for (target, region), d in frame.groupby(["target", "region"], sort=True):
        for name, column in (("v4_score", "risk_score"), ("noul", "noul_value")):
            v = pd.to_numeric(d[column], errors="coerce")
            rows.append({"cohort":cohort, "target":target, "region":region, "output":name,
                         "n":len(v), "mean":float(v.mean()), "median":float(v.median()),
                         "p10":float(v.quantile(.1)), "p90":float(v.quantile(.9))})
    return pd.DataFrame(rows)


def plot_outputs(frame, out: Path):
    targets = sorted(frame.target.unique())
    colors = {"normal":"#4c78a8", "boundary":"#f2a541", "near_failure":"#d45050"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True, sharey=True)
    for ax, target in zip(axes, targets):
        d = frame.loc[frame.target == target]
        for region in REGIONS:
            part = d.loc[d.region == region]
            ax.scatter(part.risk_score, part.noul_value, s=32, alpha=.72,
                       color=colors[region], label=region)
        ax.axhspan(N_LOW, N_HIGH, color="gray", alpha=.12)
        ax.set(title=target.replace("ttf_flowcool_", "").replace("_seconds", ""),
               xlabel="v4 Score / 2 (decision support)", xlim=(0, 1), ylim=(0, 1))
    axes[0].set_ylabel("Noul value (yes/no evidence)")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Paired Noul and fresh v4 Score; saved 05b cohort reused")
    fig.tight_layout()
    fig.savefig(out / "noul_score_scatter.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), sharey="row")
    for j, target in enumerate(targets):
        d = frame.loc[frame.target == target]
        for i, (name, column) in enumerate((("v4 Score / 2", "risk_score"), ("Noul", "noul_value"))):
            groups = [d.loc[d.region == region, column].to_numpy(dtype=float) for region in REGIONS]
            axes[i, j].boxplot(groups, tick_labels=REGIONS, patch_artist=False, showfliers=False)
            axes[i, j].set_ylim(0, 1)
            axes[i, j].set_title(target.replace("ttf_flowcool_", "").replace("_seconds", ""))
            if j == 0:
                axes[i, j].set_ylabel(name)
    fig.suptitle("Decision-output distributions by TTF surrogate band; reused cohort")
    fig.tight_layout()
    fig.savefig(out / "score_noul_distributions.png", dpi=170)
    plt.close(fig)
