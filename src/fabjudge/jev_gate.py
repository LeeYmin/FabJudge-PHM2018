"""Reusable, causal features and guarded JEV calls for LSTM alarm gates.

This module contains no dataset labels in request bodies. The caller must keep
split selection and all fitting on development sequences.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd


MODEL = "typesafe/jev-1.13"
URL = "https://openrouter.ai/api/alpha/decisions"
TIMEOUT_SECONDS = 60
POINT_SENSORS = ["ROTATIONSPEED", "IONGAUGEPRESSURE", "ETCHSUPPRESSORCURRENT",
                 "FLOWCOOLPRESSURE", "ETCHBEAMCURRENT", "FLOWCOOLFLOWRATE"]
SUMMARY_SPECS = [("ROTATIONSPEED", "rms"), ("IONGAUGEPRESSURE", "shape_factor"),
                 ("ETCHSUPPRESSORCURRENT", "peak_abs"),
                 ("FLOWCOOLPRESSURE", "shape_factor"), ("FLOWCOOLPRESSURE", "peak_abs"),
                 ("FLOWCOOLFLOWRATE", "peak_abs")]
CANDIDATES = POINT_SENSORS + [f"{name}__{stat}" for name, stat in SUMMARY_SPECS]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _state_float(value: float, serialization_version: str) -> float:
    """In v2, round each finite state float to 12 significant decimal digits.

    The rounded JSON number is used in the HTTP body and in its SHA-256 key.
    Legacy mode retains the original float representation for 06c replay.
    """
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Invalid numeric JEV state")
    if serialization_version == "v2":
        return float(format(number, ".12g"))
    if serialization_version != "legacy":
        raise ValueError("Unknown JEV state serialization version")
    return number


def load_api_key(root: Path):
    values = {}
    env_file = root / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                name, value = name.strip(), value.strip().strip('"').strip("'")
                if name in {"JEV_API_KEY", "OPENROUTER_API_KEY"} and value:
                    values[name] = value
    return (os.environ.get("JEV_API_KEY") or values.get("JEV_API_KEY") or
            os.environ.get("OPENROUTER_API_KEY") or values.get("OPENROUTER_API_KEY"))


def request_spec(row, question: dict, selected=(), *, include_history=True,
                 repeat_index=None, include_rf=True, serialization_version="legacy",
                 rf_field="rf_probability"):
    state = {"lstm_pred_seconds": _state_float(row["lstm_pred_seconds"], serialization_version),
             "recent_lstm_delta": _state_float(row["recent_lstm_delta"], serialization_version),
             "recent_lstm_std": _state_float(row["recent_lstm_std"], serialization_version)}
    if include_rf:
        state["rf_probability"] = _state_float(row[rf_field], serialization_version)
    if include_history:
        state["history_count"] = int(row["history_count"])
    state.update({name: _state_float(row[name], serialization_version) for name in selected})
    if not all(math.isfinite(value) for value in state.values()):
        raise ValueError("Invalid numeric JEV state")
    spec = {"endpoint": URL, "method": "POST", "timeout_seconds": TIMEOUT_SECONDS,
            "body": {"model": MODEL, "state": {"features": state},
                     "questions": {"alarm_evidence": question}}}
    if repeat_index is not None:
        spec["repeat_index"] = int(repeat_index)  # Cache identity only, never API input.
    return spec


def validate_response(raw: dict, question: dict):
    if not isinstance(raw, dict) or not str(raw.get("model", "")).startswith(MODEL):
        raise ValueError("Unexpected response model")
    if raw.get("provider") != "TypeSafe":
        raise ValueError("Unexpected response provider")
    answers = raw.get("answers")
    if not isinstance(answers, dict) or set(answers) != {"alarm_evidence"}:
        raise ValueError("Unexpected answers schema")
    answer = answers["alarm_evidence"]
    if not isinstance(answer, dict) or answer.get("type") != "score":
        raise ValueError("Unexpected response format")
    score, confidence = answer.get("score"), answer.get("confidence")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in (score, confidence)) or not 0 <= score <= 2 or not 0 <= confidence <= 1:
        raise ValueError("Missing, non-numeric, or out-of-range native score/confidence")
    probs = answer.get("probabilities")
    if not isinstance(probs, dict) or set(probs) != {"0", "1", "2"} or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
            for v in probs.values()) or not .95 <= sum(probs.values()) <= 1.05:
        raise ValueError("Invalid probabilities")
    if answer.get("legend") != {str(i): criterion for i, criterion in enumerate(question["criteria"])}:
        raise ValueError("Unexpected criterion legend")
    return float(score), float(probs["2"])


class JEVClient:
    def __init__(self, root: Path, cache: Path, max_new_calls: int,
                 *, serialization_version="legacy"):
        self.key = load_api_key(root)  # Never print or serialize.
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.max_new_calls = int(max_new_calls)
        self.new_calls = 0
        self.cache_hits = 0
        self.serialization_version = serialization_version

    def call_jev(self, row, question, selected=(), *, include_history=True,
                 repeat_index=None, include_rf=True, rf_field="rf_probability"):
        spec = request_spec(row, question, selected, include_history=include_history,
                            repeat_index=repeat_index, include_rf=include_rf,
                            serialization_version=self.serialization_version,
                            rf_field=rf_field)
        body_bytes = canonical(spec["body"]).encode("utf-8")
        if self.serialization_version == "v2":
            hash_input = body_bytes
            assert hash_input == body_bytes
            cache_dir = self.cache if repeat_index is None else self.cache / f"repeat_{int(repeat_index)}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            path = cache_dir / (hashlib.sha256(hash_input).hexdigest() + ".json")
        else:
            path = self.cache / (hashlib.sha256(canonical(spec).encode("utf-8")).hexdigest() + ".json")
        if path.is_file():
            self.cache_hits += 1
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("request_sha256") != path.stem:
                    raise ValueError("Cache request hash mismatch")
                score, probability_2 = validate_response(record["response"], question)
                return {"status": "cached", "score": score, "probability_2": probability_2,
                        "response": record["response"], "error": None, "cache_hit": True}
            except (KeyError, TypeError, ValueError) as exc:
                return {"status": "failed", "score": None, "probability_2": None,
                        "response": None, "error": f"Invalid cache: {type(exc).__name__}: {exc}",
                        "cache_hit": True}
        if not self.key:
            return {"status": "failed", "score": None, "probability_2": None,
                    "response": None, "error": "OpenRouter credential unavailable", "cache_hit": False}
        if self.new_calls >= self.max_new_calls:
            raise RuntimeError("New API call cap reached")
        req = urllib.request.Request(URL, data=body_bytes,
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
            method="POST")
        self.new_calls += 1
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
                response_raw = json.loads(response.read().decode("utf-8"))
            raw = {name: response_raw.get(name) for name in ("model", "provider", "answers", "usage")}
            if self.key in canonical(raw):
                raise ValueError("Credential appeared in response; refusing to cache")
            score, probability_2 = validate_response(raw, question)
            status, error = "success", None
        except urllib.error.HTTPError as exc:
            raw, score, probability_2, status, error = None, None, None, "failed", f"HTTP {exc.code}"
        except Exception as exc:
            raw, score, probability_2, status = None, None, None, "failed"
            error = f"{type(exc).__name__}: {str(exc).replace(self.key, '[REDACTED]')}"
        if status == "success":
            path.write_text(json.dumps({"request_sha256": path.stem, "response": raw,
                "latency_seconds": time.perf_counter() - started}, ensure_ascii=False, indent=2),
                encoding="utf-8")
        return {"status": status, "score": score, "probability_2": probability_2,
                "response": raw, "error": error, "cache_hit": False}


def causal_sensor_features(raw_x: np.ndarray, current: int, input_columns: list[str]):
    """Twelve 06b candidates using the current measurement and preceding 99."""
    if current < 99:
        raise ValueError("100 past/current observations unavailable")
    trailing = raw_x[current - 99:current + 1]
    result = {sensor: float(raw_x[current, input_columns.index(sensor)])
              for sensor in POINT_SENSORS}
    for sensor, statistic in SUMMARY_SPECS:
        values = trailing[:, input_columns.index(sensor)].astype(float)
        rms = float(np.sqrt(np.mean(values ** 2)))
        if statistic == "rms":
            value = rms
        elif statistic == "peak_abs":
            value = float(np.max(np.abs(values)))
        else:
            mean_abs = float(np.mean(np.abs(values)))
            value = rms / mean_abs if mean_abs > 0 else np.nan
        result[f"{sensor}__{statistic}"] = value
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError("Non-finite sensor candidate value")
    return result


def recent_history(predicted: np.ndarray, position: int):
    values = np.asarray(predicted[max(0, position - 4):position + 1], dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Non-finite LSTM history")
    return {"recent_lstm_delta": float(values[-1] - values[0]),
            "recent_lstm_std": float(np.std(values, ddof=0)),
            "history_count": int(len(values))}


def effect_size(tp, fp):
    tp, fp = np.asarray(tp, dtype=float), np.asarray(fp, dtype=float)
    tq1, tm, tq3 = np.quantile(tp, [.25, .5, .75])
    fq1, fm, fq3 = np.quantile(fp, [.25, .5, .75])
    scale = math.sqrt(((tq3 - tq1) ** 2 + (fq3 - fq1) ** 2) / 2)
    return (float((tm - fm) / scale) if scale > 0 else np.nan,
            float(tm), float(fm), float(tq3 - tq1), float(fq3 - fq1))


def metrics(name: str, predicted, truth):
    p, y = pd.Series(predicted).reset_index(drop=True), pd.Series(truth).reset_index(drop=True).astype(bool)
    missing = int(p.isna().sum())
    result = {"system": name, "evaluation_rows": len(y), "unresolved_rows": missing}
    if missing:
        return {**result, "TP": None, "FP": None, "FN": None, "TN": None,
                "precision": None, "recall": None}
    p = p.astype(bool)
    tp, fp = int((p & y).sum()), int((p & ~y).sum())
    fn, tn = int((~p & y).sum()), int((~p & ~y).sum())
    return {**result, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}


def infer_causal_sampled_rows(root: Path, sequence_table: pd.DataFrame, ids: list[str],
                              rf_bundle: dict, lstm_model, input_columns: list[str]):
    """Recreate 04b endpoints, then select 06a percentiles per sequence.

    Only x, time and raw_row_index are loaded from NPZ. Labels are derived
    afterward from the independent event time in sequence metadata.
    """
    import torch
    from .huang2018 import sampled_raw_indices
    from .huang2018_models import _rf_matrix_encoded

    lookup = sequence_table.set_index("sequence_id")
    all_rows, selected_rows = [], []
    lstm_model.eval()
    for sid in ids:
        meta = lookup.loc[sid]
        source = (root / str(meta["sequence_path"]).replace("\\", "/")).resolve()
        if not source.is_relative_to((root / "artifacts/huang2018/sequences").resolve()):
            raise ValueError("Sequence path outside expected artifacts")
        with np.load(source) as data:
            x, times, raw_index = data["x"], data["time"], data["raw_row_index"]
        if x.shape[1] != len(input_columns) or not np.all(np.diff(raw_index) > 0):
            raise ValueError("Unexpected sequence input layout")
        endpoints = np.unique(np.r_[np.arange(0, len(x), 15), len(x)-1]).astype(np.int64)
        x_at_end = x[endpoints]
        probs = rf_bundle["classifier"].predict_proba(
            _rf_matrix_encoded(x_at_end, rf_bundle["residual_model"]))[:, 1]
        predicted_samples = np.empty(len(endpoints), dtype=np.float64)
        for start in range(0, len(endpoints), 64):
            chosen = endpoints[start:start + 64]
            batch = np.zeros((len(chosen), 300, 21), dtype=np.float32)
            for j, endpoint in enumerate(chosen):
                indices = sampled_raw_indices(int(endpoint), max_length=300, sample_rate=15)
                if np.any(indices > endpoint) or np.any(times[indices] > times[endpoint]):
                    raise ValueError("Future observation in LSTM window")
                batch[j, -len(indices):] = x[indices]
            with torch.no_grad():
                predicted_samples[start:start + len(chosen)] = (
                    lstm_model(torch.from_numpy(batch))[:, -1].numpy().astype(np.float64) * 4515.0)
        seconds = predicted_samples * 4.0
        full = pd.DataFrame({"sequence_id": sid, "raw_row_index": raw_index[endpoints],
                             "lstm_pred_seconds": seconds, "rf_probability": probs})
        all_rows.append(full)
        for percentile in range(5, 100, 10):
            pos = math.floor((percentile / 100) * (len(endpoints) - 1))
            endpoint = int(endpoints[pos])
            row = {"sequence_id": sid, "raw_row_index": int(raw_index[endpoint]),
                   "sample_position": f"{percentile}%", "time": int(times[endpoint]),
                   "wall_ttf_seconds": int(meta["fault_time"]) - int(times[endpoint]),
                   "lstm_pred_seconds": float(seconds[pos]),
                   "rf_probability": float(probs[pos])}
            row.update(recent_history(seconds, pos))
            row.update(causal_sensor_features(x, endpoint, input_columns))
            selected_rows.append(row)
    return pd.DataFrame(selected_rows), pd.concat(all_rows, ignore_index=True)
