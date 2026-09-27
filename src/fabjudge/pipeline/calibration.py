"""Dev-only isotonic calibration for router and judge queue scores."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_isotonic_probability(scores, labels) -> IsotonicRegression:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels) or len(scores) < 2:
        raise ValueError("calibration arrays must be aligned one-dimensional vectors")
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("calibration inputs must be finite with binary labels")
    if len(np.unique(labels)) < 2:
        raise ValueError("isotonic calibration requires both label classes")
    model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    return model.fit(scores, labels)
