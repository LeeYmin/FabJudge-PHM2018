"""Schemas and field provenance for EvidencePacket and the LLM judge."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from jsonschema import Draft202012Validator

from .config import PACKET_VERSION

SENSOR_FIELDS = [
    "ROTATIONSPEED", "IONGAUGEPRESSURE", "ETCHSUPPRESSORCURRENT",
    "FLOWCOOLPRESSURE", "ETCHBEAMCURRENT", "FLOWCOOLFLOWRATE",
    "ROTATIONSPEED__rms", "IONGAUGEPRESSURE__shape_factor",
    "ETCHSUPPRESSORCURRENT__peak_abs", "FLOWCOOLPRESSURE__shape_factor",
    "FLOWCOOLPRESSURE__peak_abs", "FLOWCOOLFLOWRATE__peak_abs",
]
PACKET_FIELDS = [
    "lstm_pred_seconds", "recent_lstm_delta", "recent_lstm_std",
    "rf_probability", *SENSOR_FIELDS, "fault_type",
]

FIELD_METADATA = {
    "lstm_pred_seconds": {
        "description": "Causal LSTM time-to-failure prediction; smaller values are more imminent.",
        "source_artifact": "artifacts/06e_fix_and_weakness_diagnosis/long_window_features.parquet:lstm_pred_seconds",
        "calculation_id": "06e_causal_lstm_endpoint_seconds_v1",
        "unit": "seconds",
    },
    "recent_lstm_delta": {
        "description": "Current LSTM prediction minus the oldest of the last five causal endpoint predictions.",
        "source_artifact": "artifacts/06e_fix_and_weakness_diagnosis/long_window_features.parquet:lstm_pred_seconds",
        "calculation_id": "causal_last5_delta_current_minus_oldest_v1",
        "unit": "seconds",
    },
    "recent_lstm_std": {
        "description": "Population standard deviation of the last five causal endpoint LSTM predictions.",
        "source_artifact": "artifacts/06e_fix_and_weakness_diagnosis/long_window_features.parquet:lstm_pred_seconds",
        "calculation_id": "causal_last5_population_std_v1",
        "unit": "seconds",
    },
    "rf_probability": {
        "description": "Sequence-group out-of-fold RF probability for a fault within 5,000 seconds.",
        "source_artifact": "artifacts/06e_fix_and_weakness_diagnosis/long_window_features.parquet:rf_probability_oof",
        "calculation_id": "06e_sequence_group_oof_rf_probability_v1",
        "unit": None,
    },
    "fault_type": {
        "description": "Fault-family category: 1=fault1, 2=fault2, 3=fault3; not a time-to-failure label.",
        "source_artifact": "artifacts/06e_fix_and_weakness_diagnosis/long_window_features.parquet:fault",
        "calculation_id": "fault_family_code_1_2_3_v1",
        "unit": None,
    },
}

for _field in SENSOR_FIELDS:
    if "__rms" in _field:
        _description = "RMS magnitude over the current measurement and preceding 99 measurements."
        _calc = "causal_trailing100_rms_v1"
    elif "__shape_factor" in _field:
        _description = "RMS divided by mean absolute magnitude over the current and preceding 99 measurements."
        _calc = "causal_trailing100_shape_factor_v1"
    elif "__peak_abs" in _field:
        _description = "Peak absolute magnitude over the current measurement and preceding 99 measurements."
        _calc = "causal_trailing100_peak_abs_v1"
    else:
        _description = "Current normalized model-input sensor value; physical unit was not verified."
        _calc = "causal_current_model_input_v1"
    FIELD_METADATA[_field] = {
        "description": _description,
        "source_artifact": "artifacts/huang2018/sequences/*.npz:x using verified model input columns",
        "calculation_id": _calc,
        "unit": None,
    }

PACKET_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://fabjudge.local/schemas/evidence-packet-07a-v1.json",
    "title": "FabJudge EvidencePacket 07a",
    "type": "object",
    "additionalProperties": False,
    "required": ["packet_version", "fields"],
    "properties": {
        "packet_version": {"const": PACKET_VERSION},
        "fields": {
            "type": "object",
            "additionalProperties": False,
            "required": PACKET_FIELDS,
            "properties": {
                **{name: {"type": "number"} for name in PACKET_FIELDS if name != "fault_type"},
                "fault_type": {"type": "integer", "enum": [1, 2, 3]},
            },
        },
    },
    "x-field-metadata": FIELD_METADATA,
}

LLM_OUTPUT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "FabJudge LLM review evidence 07a",
    "type": "object",
    "additionalProperties": False,
    "required": ["evidence_for", "evidence_against", "unknowns", "additional_checks",
                 "priority_score", "review_recommendation"],
    "properties": {
        "evidence_for": {"type": "array", "items": {"$ref": "#/$defs/evidence"}},
        "evidence_against": {"type": "array", "items": {"$ref": "#/$defs/evidence"}},
        "unknowns": {"type": "array", "items": {"type": "string"}},
        "additional_checks": {"type": "array", "items": {"type": "string"}},
        "priority_score": {"type": "number", "minimum": 0, "maximum": 100},
        "review_recommendation": {
            "type": "string",
            "enum": ["urgent_review", "standard_review", "low_priority_review"],
        },
    },
    "$defs": {
        "evidence": {
            "type": "object", "additionalProperties": False,
            "required": ["field", "value", "claim"],
            "properties": {
                "field": {"type": "string", "enum": PACKET_FIELDS},
                "value": {"type": "number"},
                "claim": {"type": "string"},
            },
        },
    },
}

FORBIDDEN_PACKET_TOKENS = (
    "sequence", "filename", "file_name", "filepath", "row_index", "raw_row",
    "timestamp", "time", "wall_ttf", "actual_ttf", "label", "source", "split",
    "event", "sample_position", "machine", "tool", "run_id",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def make_packet(fields: dict[str, Any]) -> dict[str, Any]:
    if set(fields) != set(PACKET_FIELDS):
        raise ValueError("EvidencePacket fields do not match the frozen schema")
    packet_fields: dict[str, Any] = {}
    for name in PACKET_FIELDS:
        value = fields[name]
        if value is None:
            packet_fields[name] = None
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"EvidencePacket field is not numeric: {name}")
        elif not math.isfinite(float(value)):
            packet_fields[name] = None
        elif name == "fault_type":
            packet_fields[name] = int(value)
        else:
            packet_fields[name] = float(value)
    return {"packet_version": PACKET_VERSION, "fields": packet_fields}


def packet_issues(packet: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    _assert_no_forbidden_keys(packet)
    if set(packet) != {"packet_version", "fields"}:
        issues.append("unexpected top-level fields")
    fields = packet.get("fields")
    if not isinstance(fields, dict) or set(fields) != set(PACKET_FIELDS):
        issues.append("field set differs from schema")
        return issues
    for name, value in fields.items():
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            issues.append(f"missing or non-numeric field: {name}")
        elif not math.isfinite(float(value)):
            issues.append(f"non-finite field: {name}")
    if fields.get("fault_type") not in {1, 2, 3}:
        issues.append("fault_type is outside categories 1, 2, 3")
    return issues


def _assert_no_forbidden_keys(value: Any, path: str = "packet") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_l = str(key).lower()
            if any(token in key_l for token in FORBIDDEN_PACKET_TOKENS):
                raise ValueError(f"Forbidden identifier/label field in packet: {path}.{key}")
            _assert_no_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_keys(child, f"{path}[{index}]")


def validate_packet(packet: dict[str, Any]) -> None:
    issues = packet_issues(packet)
    if issues:
        raise ValueError("DATA_REVIEW: " + "; ".join(issues))
    Draft202012Validator(PACKET_SCHEMA).validate(packet)


def validate_llm_output(output: Any) -> None:
    Draft202012Validator(LLM_OUTPUT_SCHEMA).validate(output)
