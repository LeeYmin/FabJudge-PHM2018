"""Alarm-preserving three-level router and deterministic sample positions."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def nested_percentiles(per_sequence_cap: int) -> list[int]:
    """Return a nested, deterministic percentile grid with at most the cap."""
    grids = {
        10: list(range(5, 100, 10)),
        5: [10, 30, 50, 70, 90],
        2: [25, 75],
        1: [50],
    }
    if per_sequence_cap not in grids:
        raise ValueError("per_sequence_cap must be one of 1, 2, 5, 10")
    return grids[per_sequence_cap]


def choose_percentile_capacity(alarm_counts: Iterable[int], total_cap: int = 600) -> list[int]:
    counts = [max(0, int(n)) for n in alarm_counts]
    for cap in (10, 5, 2, 1):
        positions = nested_percentiles(cap)
        total = sum(min(n, len(positions)) for n in counts if n > 0)
        if total <= total_cap:
            return positions
    raise RuntimeError("Sample cap cannot be met while retaining one point per alarmed sequence")


def percentile_positions(n: int, percentiles: Iterable[int]) -> list[tuple[int, int]]:
    if n <= 0:
        return []
    return [(p, math.floor((p / 100.0) * (n - 1))) for p in percentiles]


def route_score(score: float, t_low: float, t_high: float, *,
                low_audit_rate: float = 0.10, audit_draw: float = 1.0) -> dict:
    """Route every alert to a queue; the router never cancels an alarm."""
    if not all(math.isfinite(float(v)) for v in (score, t_low, t_high, low_audit_rate, audit_draw)):
        return {"route": "DATA_REVIEW", "llm_call": False, "alarm_retained": True}
    if not 0 <= low_audit_rate <= 1 or t_low >= t_high:
        raise ValueError("Invalid thresholds or LOW audit rate")
    if score >= t_high:
        return {"route": "HIGH", "llm_call": False, "alarm_retained": True}
    if score >= t_low:
        return {"route": "MID", "llm_call": True, "alarm_retained": True}
    call = audit_draw < low_audit_rate
    return {"route": "LOW", "llm_call": call, "alarm_retained": True}


def random_audit_indices(n_rows: int, budget: int, *, seed: int,
                         repetitions: int = 100) -> list[np.ndarray]:
    if n_rows < 0 or budget < 0 or budget > n_rows or repetitions <= 0:
        raise ValueError("Invalid random-router dimensions")
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(n_rows, size=budget, replace=False))
            for _ in range(repetitions)]


def compose_calibrated_queue_scores(
    llm_priority_scores: Iterable[float | None],
    router_scores: Iterable[float | None],
    llm_called: Iterable[bool],
    llm_calibrator,
    router_calibrator,
    *,
    row_statuses: Iterable[str] | None = None,
) -> list[dict]:
    """Use calibrated LLM scores when available, otherwise calibrated router scores.

    DATA_REVIEW and MODEL_UNAVAILABLE stay explicit queue states; neither is
    replaced by a numeric score.
    """
    llm_scores = list(llm_priority_scores)
    raw_router_scores = list(router_scores)
    called = list(llm_called)
    statuses = list(row_statuses) if row_statuses is not None else ["READY"] * len(called)
    if not (len(llm_scores) == len(raw_router_scores) == len(called) == len(statuses)):
        raise ValueError("queue score inputs must have identical lengths")
    result = []
    for llm_score, router_score, was_called, status in zip(llm_scores, raw_router_scores, called, statuses):
        if status == "DATA_REVIEW":
            result.append({"queue_score": None, "status": "DATA_REVIEW", "score_source": None})
            continue
        if status == "MODEL_UNAVAILABLE" or (was_called and (llm_score is None or not math.isfinite(float(llm_score)))):
            result.append({"queue_score": None, "status": "MODEL_UNAVAILABLE", "score_source": None})
            continue
        selected_score = llm_score if was_called else router_score
        if selected_score is None or not math.isfinite(float(selected_score)):
            result.append({"queue_score": None, "status": "MODEL_UNAVAILABLE", "score_source": None})
            continue
        calibrator = llm_calibrator if was_called else router_calibrator
        calibrated = float(np.asarray(calibrator.predict([float(selected_score)])).reshape(-1)[0])
        if not math.isfinite(calibrated):
            result.append({"queue_score": None, "status": "MODEL_UNAVAILABLE", "score_source": None})
            continue
        result.append({"queue_score": calibrated, "status": "READY",
                       "score_source": "llm_priority_score" if was_called else "router_score"})
    return result
