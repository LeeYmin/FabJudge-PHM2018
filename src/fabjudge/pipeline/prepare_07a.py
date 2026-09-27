"""Freeze the 07a schemas, R1 question, and preregistration before API calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import (
    ARTIFACT_DIR, HTTP_TIMEOUT_SECONDS, JEV_KEY_ENV, JEV_MODEL_ID,
    LLM_KEY_ENV, LLM_MAX_OUTPUT_TOKENS, LLM_MODEL_ID, LOW_AUDIT_RATE,
    MAX_JEV_EXTRA_CALLS, PACKET_VERSION, RANDOM_SEED, SAMPLE_CAP,
    SERIALIZATION_VERSION,
)
from .schemas import LLM_OUTPUT_SCHEMA, PACKET_SCHEMA, canonical_json, sha256_json


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_r1_question() -> dict:
    instructions = (
        "Using only state.features, evaluate the evidence associated with this existing LSTM alarm for a fault within 5000 seconds. "
        "Do not estimate a new time to failure. Weigh supporting and contradicting evidence in a balanced way. "
        "The alarm is never cancelled by this score; the score only describes how deeply the case should be analyzed. "
        "lstm_pred_seconds: smaller values indicate a more imminent fault. "
        "recent_lstm_delta: current minus the oldest of the last five causal LSTM predictions; negative means the predicted failure time is falling. "
        "recent_lstm_std: population variability across those five predictions. "
        "rf_probability: sequence-group out-of-fold estimated probability of a fault within 5000 seconds. "
        "ROTATIONSPEED, IONGAUGEPRESSURE, ETCHSUPPRESSORCURRENT, FLOWCOOLPRESSURE, ETCHBEAMCURRENT, and FLOWCOOLFLOWRATE "
        "are current normalized model-input sensor readings; their physical units are unverified. "
        "ROTATIONSPEED__rms is RMS magnitude over the current and preceding 99 readings. "
        "IONGAUGEPRESSURE__shape_factor and FLOWCOOLPRESSURE__shape_factor are RMS divided by mean absolute magnitude over those 100 readings. "
        "ETCHSUPPRESSORCURRENT__peak_abs, FLOWCOOLPRESSURE__peak_abs, and FLOWCOOLFLOWRATE__peak_abs are peak absolute magnitude over those 100 readings. "
        "fault_type identifies the fault family (1, 2, or 3); it is not evidence of when the fault will occur. "
        "Choose Weak, Uncertain, or Strong from the combined evidence. Do not infer sequence identity, timestamp, or actual failure time."
    )
    return {
        "type": "score",
        "instructions": instructions,
        "criteria": [
            "Weak: the LSTM trend, RF signal, and sensor evidence give weak support for urgent analysis; the alarm remains in the review queue",
            "Uncertain: the LSTM trend, RF signal, and sensor evidence are mixed or insufficient; the alarm remains in the review queue",
            "Strong: the LSTM trend, RF signal, and sensor evidence give strong support for deeper analysis; the alarm remains in the review queue",
        ],
    }


def build_config() -> dict:
    return {
        "phase": "07a",
        "packet_version": PACKET_VERSION,
        "serialization_version": SERIALIZATION_VERSION,
        "jev_key_env": JEV_KEY_ENV,
        "llm_key_env": LLM_KEY_ENV,
        "jev_model_id": JEV_MODEL_ID,
        "llm_model_id": LLM_MODEL_ID,
        "llm_model_id_source": "artifacts/jev_tut/excluded_gpt_luna/config.json",
        "llm_catalog_url": "https://openrouter.ai/api/v1/models",
        "llm_chat_url": "https://openrouter.ai/api/v1/chat/completions",
        "catalog_must_confirm_exact_id_before_inference": True,
        "llm_generation": {
            "temperature": 0,
            "reasoning_effort": "none",
            "max_tokens": LLM_MAX_OUTPUT_TOKENS,
            "response_format": "strict_json_schema",
        },
        "low_audit_rate": LOW_AUDIT_RATE,
        "dev_sample_cap": SAMPLE_CAP,
        "jev_extra_stability_call_cap": MAX_JEV_EXTRA_CALLS,
        "llm_smoke_calls": 10,
        "http_timeout_seconds": HTTP_TIMEOUT_SECONDS,
        "random_seed": RANDOM_SEED,
        "test_sequence_use": "07c exactly once; test sequences excluded from every 07a/07b fit and call",
        "credential_policy": "role-specific exact env names; no cross-role or generic fallback; key values never recorded",
    }


def build_preregistration(out_dir: Path = ARTIFACT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "packet_schema.json", PACKET_SCHEMA)
    _write_json(out_dir / "llm_output_schema.json", LLM_OUTPUT_SCHEMA)
    _write_json(out_dir / "questions" / "R1.json", build_r1_question())
    config = build_config()
    _write_json(out_dir / "config.json", config)

    prereg = {
        "protocol": "FabJudge 07 preregistration",
        "version": "prereg_07_v1",
        "phase": "07a",
        "created_before_07b_results": True,
        "1_evaluation_unit_and_sampling": {
            "population": "all train plus validation sequences for fault1, fault2, fault3; no test sequences",
            "unit": "causal endpoint with LSTM prediction <= 5000 seconds, sampled within original sequence",
            "alarm_definition": "lstm_pred_seconds <= 5000 seconds; TTF label is not used for selection",
            "initial_percentiles": list(range(5, 100, 10)),
            "position_formula": "floor((percentile/100)*(n_alarm_endpoints-1))",
            "cap": SAMPLE_CAP,
            "nested_reduction": {
                "10": list(range(5, 100, 10)),
                "5": [10, 30, 50, 70, 90],
                "2": [25, 75],
                "1": [50],
                "rule": "use the largest per-sequence grid whose total selected rows is <=600; retain one median row for each alarmed sequence",
            },
            "dev_label": "actual TTF <= 5000 seconds; evaluation only, stored in dev_labels.csv",
            "grouping": "original sequence; exported group identifiers are opaque hashes and never enter a model packet",
            "rf": "06e sequence-group OOF rf_probability_oof",
            "caveats": [
                "train LSTM predictions are in-sample; LSTM-prediction ranking can be optimistic on train development sequences",
                "sampling reduces repeated within-sequence endpoints and does not estimate event-rate prevalence",
            ],
        },
        "2_evidence_packet": {
            "packet_version": PACKET_VERSION,
            "fields": ["lstm_pred_seconds", "recent_lstm_delta", "recent_lstm_std", "rf_probability",
                       "ROTATIONSPEED", "IONGAUGEPRESSURE", "ETCHSUPPRESSORCURRENT", "FLOWCOOLPRESSURE",
                       "ETCHBEAMCURRENT", "FLOWCOOLFLOWRATE", "ROTATIONSPEED__rms",
                       "IONGAUGEPRESSURE__shape_factor", "ETCHSUPPRESSORCURRENT__peak_abs",
                       "FLOWCOOLPRESSURE__shape_factor", "FLOWCOOLPRESSURE__peak_abs",
                       "FLOWCOOLFLOWRATE__peak_abs", "fault_type"],
            "missing_or_nonfinite": "DATA_REVIEW; no imputation and no JEV/LLM request",
            "causality": "only current/past model inputs and preceding 100 readings; no future rows or samples_to_fault",
            "excluded_fields": ["sequence_id", "file name", "source file", "timestamp", "raw row index", "TTF", "label", "split", "router output", "JEV score"],
            "packet_schema_sha256": _hash_file(out_dir / "packet_schema.json"),
            "field_metadata": "packet_schema.json x-field-metadata contains one-line descriptions, source artifacts, calculation IDs, and confirmed units only",
        },
        "3_jev_router": {
            "question_file": "questions/R1.json",
            "question_sha256": _hash_file(out_dir / "questions" / "R1.json"),
            "basis": "06c P1 balanced evidence wording; no fault3 TP/FP median or midpoint; descriptions of all EvidencePacket fields",
            "score": "native score in [0,2] and probabilities; branch on native score",
            "routes": {
                "HIGH": "score >= t_high; standard review queue; no LLM",
                "MID": "t_low <= score < t_high; full LLM judge",
                "LOW": "score < t_low; low-priority review queue; random audit only",
                "alarm_policy": "every alarm remains in a human review queue; no score can cancel it",
                "low_audit_rate": LOW_AUDIT_RATE,
                "threshold_selection": "dev only; select t_low and t_high to meet each 30%, 50%, 70% LLM-call budget with the fixed audit rate; no threshold values or outcome-based decision in 07a",
            },
            "comparators": {
                "rf": "same three-way rule using rf_probability; dev thresholds match each JEV call budget",
                "random": "exact same number of alert samples sent to LLM, selected without replacement; seed fixed; 100 repetitions",
            },
        },
        "4_llm_judge": {
            "model_id": LLM_MODEL_ID,
            "model_source": "previous exact configured ID from artifacts/jev_tut/excluded_gpt_luna/config.json; current catalog verification required before calls",
            "auth_role": "llm",
            "generation": build_config()["llm_generation"],
            "request_data": "EvidencePacket only; no route, JEV score, or separate RF-router field",
            "output_schema_sha256": _hash_file(out_dir / "llm_output_schema.json"),
            "citation_tolerance": {"absolute": 1e-6, "relative": 1e-6, "rule": "abs(cited-packet) <= abs_tol + rel_tol*abs(packet)"},
            "invalid_schema": "MODEL_UNAVAILABLE; no numeric fallback or zero imputation",
            "smoke": {"sample_n": 10, "seed": RANDOM_SEED, "label_blind": True, "minimum_schema_passes": 8},
        },
        "5_queue_and_metrics": {
            "queue_score": "LLM-called valid row uses LLM priority_score; uncalled row uses the router score; each score source is isotonic-calibrated on dev to estimated P(TTF<=5000)",
            "calibration": {
                "method": "sklearn IsotonicRegression(increasing=True, y_min=0, y_max=1, out_of_bounds='clip')",
                "fit_data": "all valid sampled dev rows; fit separately by fault and score source; LLM calibrator uses the full-dev LLM outputs generated once in 07b; no test rows",
                "sources": ["LLM priority_score", "JEV native score", "RF OOF probability"],
            },
            "primary_metric": "precision@K at K=round-half-up(10% of valid dev alarm samples); report fault1, fault2, and pooled fault1+fault2 as primary result",
            "secondary_metrics": [
                "LLM-call ratio and cost including JEV cost",
                "LOW audit sample near-failure fraction",
                "LLM evidence-claim citation agreement rate",
                "precision@K at 5% and 20%",
            ],
            "baselines": {
                "no_llm_lstm": "ascending lstm_pred_seconds",
                "no_llm_rf": "descending rf_probability OOF",
                "no_llm_logistic": "numeric EvidencePacket fields; StandardScaler then LogisticRegression(C=1, solver='lbfgs', max_iter=5000); GroupKFold(6) by original sequence for OOF predictions",
                "full_llm": "all valid sampled dev packets called once; results reused for routed comparisons",
            },
            "bootstrap": {
                "method": "paired sequence-cluster bootstrap, sampling sequences with replacement within fault and retaining all sampled rows",
                "repetitions": 5000,
                "seed": RANDOM_SEED,
                "interval": "percentile 95% CI",
            },
        },
        "6_success_criteria_and_stopping": {
            "decision_budget": 0.50,
            "role_recognition": {
                "noninferiority": "JEV-router precision@10%-K minus full-LLM precision@10%-K; sequence-bootstrap 95% CI lower bound >= -0.05",
                "random_superiority": "JEV-router minus random-router precision@10%-K; sequence-bootstrap 95% CI lower bound > 0",
            },
            "jev_advantage": "at same 50% LLM budget, JEV-router minus RF-router precision@10%-K; CI lower bound > 0",
            "llm_value_separate": "full-LLM minus logistic-baseline precision@10%-K with CI; not part of JEV decision",
            "additional_budgets": "30% and 70% reported; no success/failure decision at those budgets",
            "failure_wording": "JEV 라우팅의 호출 절감 효과 미확인",
            "no_posthoc_relaxation": True,
            "stop_conditions": [
                "packet label/identifier leakage", "configured model absent from catalog", "first JEV or LLM response schema failure",
                "fewer than 8 of 10 LLM smoke outputs pass schema", "call cap reached", "preregistration hash mismatch",
            ],
            "api_limits": {"jev_new_calls": "number of READY dev samples + 15 stability repeats", "jev_stability": "5 rows x 3 additional calls", "llm_smoke_inference": 10},
            "test_policy": "all test sequences for fault1/fault2/fault3 are untouched until one evaluation in 07c",
        },
        "artifacts": {
            "config_sha256": _hash_file(out_dir / "config.json"),
            "llm_output_schema_sha256": _hash_file(out_dir / "llm_output_schema.json"),
        },
    }
    prereg["sha256"] = sha256_json(prereg)
    _write_json(out_dir / "prereg_07.json", prereg)
    return prereg


def verify_preregistration(path: Path) -> str:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    supplied = value.pop("sha256", None)
    computed = sha256_json(value)
    if supplied != computed:
        raise RuntimeError("prereg_07.json hash mismatch; stopping before API calls")
    return computed
