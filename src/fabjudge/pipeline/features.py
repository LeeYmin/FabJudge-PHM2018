"""Deterministic causal features matching the 06b twelve-sensor candidates."""

from __future__ import annotations

import math

import numpy as np

POINT_SENSORS = [
    "ROTATIONSPEED", "IONGAUGEPRESSURE", "ETCHSUPPRESSORCURRENT",
    "FLOWCOOLPRESSURE", "ETCHBEAMCURRENT", "FLOWCOOLFLOWRATE",
]
SUMMARY_SPECS = [
    ("ROTATIONSPEED", "rms"),
    ("IONGAUGEPRESSURE", "shape_factor"),
    ("ETCHSUPPRESSORCURRENT", "peak_abs"),
    ("FLOWCOOLPRESSURE", "shape_factor"),
    ("FLOWCOOLPRESSURE", "peak_abs"),
    ("FLOWCOOLFLOWRATE", "peak_abs"),
]


def causal_sensor_features(x: np.ndarray, position: int, input_columns: list[str]) -> dict[str, float]:
    if position < 99:
        raise ValueError("100 current/past sensor observations are unavailable")
    if x.ndim != 2 or x.shape[1] != len(input_columns):
        raise ValueError("Unexpected model-input matrix layout")
    trailing = np.asarray(x[position - 99:position + 1], dtype=np.float64)
    result = {name: float(x[position, input_columns.index(name)]) for name in POINT_SENSORS}
    for sensor, statistic in SUMMARY_SPECS:
        values = trailing[:, input_columns.index(sensor)]
        rms = float(np.sqrt(np.mean(np.square(values))))
        if statistic == "rms":
            value = rms
        elif statistic == "peak_abs":
            value = float(np.max(np.abs(values)))
        elif statistic == "shape_factor":
            mean_abs = float(np.mean(np.abs(values)))
            value = rms / mean_abs if mean_abs > 0 else float("nan")
        else:
            raise ValueError(f"Unknown summary statistic: {statistic}")
        result[f"{sensor}__{statistic}"] = value
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("Non-finite sensor candidate")
    return result


def recent_lstm_features(predictions: np.ndarray, endpoint_position: int) -> dict[str, float]:
    values = np.asarray(predictions[max(0, endpoint_position - 4):endpoint_position + 1], dtype=np.float64)
    if len(values) != 5 or not np.isfinite(values).all():
        raise ValueError("Five finite causal LSTM endpoints are unavailable")
    return {
        "recent_lstm_delta": float(values[-1] - values[0]),
        "recent_lstm_std": float(np.std(values, ddof=0)),
    }
