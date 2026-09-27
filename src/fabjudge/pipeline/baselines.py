"""Deterministic no-LLM ranking baselines for development evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schemas import PACKET_FIELDS, validate_packet

NUMERIC_FIELDS = [name for name in PACKET_FIELDS if name != "fault_type"]


def packet_feature_frame(packets: list[dict]) -> pd.DataFrame:
    rows = []
    for packet in packets:
        validate_packet(packet)
        rows.append(packet["fields"])
    frame = pd.DataFrame(rows, columns=PACKET_FIELDS)
    if frame.empty:
        raise ValueError("Cannot build a baseline matrix from no packets")
    if not np.isfinite(frame[NUMERIC_FIELDS].to_numpy(dtype=np.float64)).all():
        raise ValueError("DATA_REVIEW packets cannot enter a numeric baseline")
    if not frame["fault_type"].isin([1, 2, 3]).all():
        raise ValueError("Unexpected fault category")
    return frame


def logistic_group_oof(
    packets: list[dict], labels, sequence_groups, *, n_splits: int = 6,
) -> pd.DataFrame:
    """Return GroupKFold OOF probabilities; every validation sequence is held out."""
    frame = packet_feature_frame(packets)
    y = np.asarray(labels, dtype=np.int8)
    groups = np.asarray(sequence_groups, dtype=str)
    if len(y) != len(frame) or len(groups) != len(frame):
        raise ValueError("packet, label, and sequence group lengths differ")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("logistic baseline requires both binary label classes")
    if len(np.unique(groups)) < n_splits:
        raise ValueError("not enough unique sequence groups for GroupKFold")
    probability = np.full(len(frame), np.nan, dtype=np.float64)
    fold_id = np.full(len(frame), -1, dtype=np.int16)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train, test) in enumerate(splitter.split(frame, y, groups), start=1):
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("GroupKFold split a sequence across train and validation")
        if len(np.unique(y[train])) < 2:
            raise ValueError(f"logistic baseline fold {fold} lacks one label class")
        fold_model = Pipeline([
            ("features", ColumnTransformer([
                ("numeric", StandardScaler(), NUMERIC_FIELDS),
                ("fault", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["fault_type"]),
            ])),
            ("logistic", LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000)),
        ])
        fold_model.fit(frame.iloc[train], y[train])
        probability[test] = fold_model.predict_proba(frame.iloc[test])[:, 1]
        fold_id[test] = fold
    if not np.isfinite(probability).all() or np.any(fold_id < 1):
        raise AssertionError("Some dev rows have no logistic OOF prediction")
    return pd.DataFrame({"logistic_probability_oof": probability, "group_fold": fold_id})
