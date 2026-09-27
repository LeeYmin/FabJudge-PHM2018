"""Pre-registered review-capacity metrics for the 07 series."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def k_for_fraction(n_rows: int, fraction: float) -> int:
    if n_rows < 0 or not 0 < fraction <= 1:
        raise ValueError("Invalid row count or K fraction")
    return 0 if n_rows == 0 else max(1, int(math.floor(n_rows * fraction + 0.5)))


def precision_at_k(labels: Sequence[int | bool], scores: Sequence[float],
                   fraction: float = 0.10) -> dict:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s):
        raise ValueError("labels and scores must be one-dimensional and aligned")
    if not len(y):
        return {"precision_at_k": None, "k": 0, "true_positives_at_k": 0, "n": 0}
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError("labels must be binary and scores finite")
    k = k_for_fraction(len(y), fraction)
    ranked = np.argsort(-s, kind="stable")[:k]
    tp = int(y[ranked].sum())
    return {"precision_at_k": tp / k, "k": k, "true_positives_at_k": tp, "n": len(y)}


def grouped_bootstrap_precision_difference(
    labels: Sequence[int | bool], scores_a: Sequence[float], scores_b: Sequence[float],
    groups: Sequence[str], faults: Sequence[str], *, fraction: float = 0.10,
    repetitions: int = 5000, seed: int = 20260927,
) -> dict:
    """Stratified sequence bootstrap CI for precision@K(A)-precision@K(B)."""
    y = np.asarray(labels, dtype=np.int8)
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    g, f = np.asarray(groups, str), np.asarray(faults, str)
    if not (len(y) == len(a) == len(b) == len(g) == len(f)):
        raise ValueError("bootstrap inputs must have identical lengths")
    observed = precision_at_k(y, a, fraction)["precision_at_k"] - precision_at_k(y, b, fraction)["precision_at_k"]
    unique_by_fault = {fault: np.unique(g[f == fault]) for fault in np.unique(f)}
    rows_by_group = {group: np.flatnonzero(g == group) for group in np.unique(g)}
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for rep in range(repetitions):
        picked_rows = []
        for groups_in_fault in unique_by_fault.values():
            chosen = rng.choice(groups_in_fault, size=len(groups_in_fault), replace=True)
            picked_rows.extend(rows_by_group[item] for item in chosen)
        indices = np.concatenate(picked_rows) if picked_rows else np.array([], dtype=int)
        draws[rep] = (precision_at_k(y[indices], a[indices], fraction)["precision_at_k"] -
                      precision_at_k(y[indices], b[indices], fraction)["precision_at_k"])
    return {
        "estimate": float(observed), "ci_95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "bootstrap_repetitions": repetitions, "seed": seed,
        "resampling_unit": "sequence_group, sampled with replacement within fault",
    }
