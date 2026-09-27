"""Credential-separated, cached OpenRouter clients for JEV and the LLM judge."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request
from typing import Any

import requests

from .config import (
    HTTP_TIMEOUT_SECONDS, JEV_KEY_ENV, JEV_MODEL_ID, JEV_URL,
    LLM_KEY_ENV, LLM_MODEL_ID, OPENROUTER_CHAT_URL, OPENROUTER_MODELS_URL,
    SERIALIZATION_VERSION,
)
from .keys import load_jev_key, load_llm_key
from .schemas import (
    LLM_OUTPUT_SCHEMA, canonical_json, sha256_json, validate_llm_output,
    validate_packet,
)


def _has_secret(value: Any, secret: str) -> bool:
    if isinstance(value, str):
        return bool(secret) and secret in value
    if isinstance(value, dict):
        return any(_has_secret(k, secret) or _has_secret(v, secret) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_secret(item, secret) for item in value)
    return False


def _safe_error(exc: Exception, secret: str) -> str:
    message = str(exc)
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return f"{type(exc).__name__}: {message[:500]}"


def _cost_from_usage(usage: Any) -> float | None:
    if not isinstance(usage, dict):
        return None
    for key in ("cost", "total_cost"):
        value = usage.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            return float(value)
    return None


def _safe_cache_write(path: Path, record: dict, key: str) -> None:
    if _has_secret(record, key):
        raise ValueError("Credential appeared in cache record; refusing to persist")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class JEVClient:
    """JEV-only client. It has no constructor parameter that accepts a key."""

    def __init__(self, root: Path, cache_dir: Path, max_new_calls: int):
        self._key = load_jev_key(root)
        self.cache_dir = Path(cache_dir)
        self.max_new_calls = int(max_new_calls)
        self.new_calls = 0
        self.cache_hits = 0
        self.attempts: list[dict] = []

    def call(self, packet: dict, question: dict, *, repeat_index: int | None = None) -> dict:
        validate_packet(packet)
        body = {
            "model": JEV_MODEL_ID,
            "state": {"features": packet["fields"]},
            "questions": {"alarm_evidence": question},
        }
        body_bytes = canonical_json(body).encode("utf-8")
        digest = hashlib.sha256(body_bytes).hexdigest()
        base_dir = self.cache_dir if repeat_index is None else self.cache_dir / f"repeat_{int(repeat_index)}"
        cache_path = base_dir / f"{digest}.json"
        if cache_path.is_file():
            self.cache_hits += 1
            try:
                record = json.loads(cache_path.read_text(encoding="utf-8"))
                if record.get("request_sha256") != digest or record.get("key_role") != "jev":
                    raise ValueError("Cache identity mismatch")
                result = self._validated_result(record["response"], True, 0.0, question)
                self.attempts.append(self._attempt_row(digest, result))
                return result
            except Exception as exc:
                result = {"status": "MODEL_UNAVAILABLE", "error": _safe_error(exc, self._key),
                          "score": None, "probabilities": None, "cache_hit": True}
                self.attempts.append(self._attempt_row(digest, result))
                return result
        if self.new_calls >= self.max_new_calls:
            raise RuntimeError("JEV call cap reached")
        request = urllib.request.Request(
            JEV_URL, data=body_bytes,
            headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
            method="POST",
        )
        self.new_calls += 1
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                response_text = response.read().decode("utf-8")
            if self._key in response_text:
                result = {"status": "MODEL_UNAVAILABLE", "error": "Credential appeared as [REDACTED]; refusing to cache",
                          "score": None, "probabilities": None, "cache_hit": False,
                          "latency_seconds": time.perf_counter() - started, "credential_leak": True}
                self.attempts.append(self._attempt_row(digest, result))
                return result
            decoded = json.loads(response_text)
            response_payload = {name: decoded.get(name) for name in ("model", "provider", "answers", "usage")}
            result = self._validated_result(response_payload, False, time.perf_counter() - started, question)
            if result["status"] == "success":
                _safe_cache_write(cache_path, {
                    "serialization_version": SERIALIZATION_VERSION,
                    "request_sha256": digest,
                    "key_role": "jev", "key_env_var": JEV_KEY_ENV,
                    "model_id": JEV_MODEL_ID, "response": response_payload,
                    "latency_seconds": result["latency_seconds"],
                }, self._key)
        except urllib.error.HTTPError as exc:
            result = {"status": "MODEL_UNAVAILABLE", "error": f"HTTP {exc.code}",
                      "score": None, "probabilities": None, "cache_hit": False,
                      "latency_seconds": time.perf_counter() - started}
        except Exception as exc:
            result = {"status": "MODEL_UNAVAILABLE", "error": _safe_error(exc, self._key),
                      "score": None, "probabilities": None, "cache_hit": False,
                      "latency_seconds": time.perf_counter() - started}
        self.attempts.append(self._attempt_row(digest, result))
        return result

    def _validated_result(self, raw: dict, cache_hit: bool, latency: float, question: dict) -> dict:
        try:
            if not isinstance(raw, dict) or not str(raw.get("model", "")).startswith(JEV_MODEL_ID):
                raise ValueError("Unexpected JEV response model")
            if raw.get("provider") != "TypeSafe":
                raise ValueError("Unexpected JEV response provider")
            answers = raw.get("answers")
            if not isinstance(answers, dict) or set(answers) != {"alarm_evidence"}:
                raise ValueError("Unexpected JEV answers schema")
            answer = answers["alarm_evidence"]
            if not isinstance(answer, dict) or answer.get("type") != "score":
                raise ValueError("Unexpected JEV score format")
            score = answer.get("score")
            probs = answer.get("probabilities")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 2:
                raise ValueError("Invalid JEV native score")
            if not isinstance(probs, dict) or set(probs) != {"0", "1", "2"}:
                raise ValueError("Invalid JEV probabilities")
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in probs.values()):
                raise ValueError("Invalid JEV probability value")
            if not 0.95 <= sum(probs.values()) <= 1.05:
                raise ValueError("JEV probabilities do not sum to one")
            if answer.get("legend") != {str(i): criterion for i, criterion in enumerate(question["criteria"])}:
                raise ValueError("JEV criterion legend does not match R1")
            cost = _cost_from_usage(raw.get("usage"))
            return {"status": "success", "score": float(score),
                    "probabilities": {str(k): float(v) for k, v in probs.items()},
                    "confidence": answer.get("confidence"), "cost_usd": cost,
                    "latency_seconds": float(latency), "cache_hit": cache_hit,
                    "error": None}
        except Exception as exc:
            return {"status": "MODEL_UNAVAILABLE", "score": None, "probabilities": None,
                    "cost_usd": _cost_from_usage(raw.get("usage") if isinstance(raw, dict) else None),
                    "latency_seconds": float(latency), "cache_hit": cache_hit,
                    "error": _safe_error(exc, self._key)}

    @staticmethod
    def _attempt_row(digest: str, result: dict) -> dict:
        return {"request_sha256": digest, "key_role": "jev", "key_env_var": JEV_KEY_ENV,
                "model_id": JEV_MODEL_ID, "status": result["status"],
                "cache_hit": bool(result.get("cache_hit")),
                "latency_seconds": result.get("latency_seconds"),
                "cost_usd": result.get("cost_usd"), "error": result.get("error")}


class LLMClient:
    """Luna-only client for catalog verification and evidence judge calls."""

    def __init__(self, root: Path, cache_dir: Path, max_new_calls: int):
        self._key = load_llm_key(root)
        self.cache_dir = Path(cache_dir)
        self.max_new_calls = int(max_new_calls)
        self.new_calls = 0
        self.cache_hits = 0
        self.attempts: list[dict] = []
        self.session = requests.Session()

    def catalog_model(self) -> dict:
        try:
            response = self.session.get(
                OPENROUTER_MODELS_URL,
                headers={"Authorization": f"Bearer {self._key}"},
                timeout=HTTP_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise RuntimeError(_safe_error(exc, self._key)) from None
        if self._key in response.text:
            raise RuntimeError("Credential appeared as [REDACTED] in model catalog response; refusing to retain it")
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter model catalog returned HTTP {response.status_code}")
        data = response.json().get("data")
        if not isinstance(data, list):
            raise RuntimeError("OpenRouter model catalog response has no model list")
        entry = next((item for item in data if isinstance(item, dict) and item.get("id") == LLM_MODEL_ID), None)
        if entry is None:
            raise RuntimeError("Configured LLM model ID was not found in the OpenRouter catalog")
        return {
            "model_id": entry["id"],
            "name": entry.get("name"),
            "pricing": entry.get("pricing"),
            "supported_parameters": entry.get("supported_parameters"),
            "top_provider": entry.get("top_provider"),
            "key_role": "llm", "key_env_var": LLM_KEY_ENV,
        }

    def call(self, packet: dict, output_schema: dict, *, model_metadata: dict) -> dict:
        validate_packet(packet)
        supported = set(model_metadata.get("supported_parameters") or [])
        if "response_format" not in supported:
            raise RuntimeError("Configured Luna model does not advertise response_format support")
        body = {
            "model": LLM_MODEL_ID,
            "messages": [
                {"role": "system", "content": (
                    "You are a review-priority judge for an existing semiconductor process alarm. "
                    "The alarm must remain in a human review queue; never cancel or suppress it. "
                    "Use only the EvidencePacket in the user message. Weigh evidence for and against. "
                    "Do not infer time, sequence identity, labels, or information absent from the packet. "
                    "Cite only exact packet field names and numeric values. priority_score is 0 to 100 and "
                    "means strength of evidence for failure within 5000 seconds. Return the required JSON object."
                )},
                {"role": "user", "content": canonical_json(packet)},
            ],
            "temperature": 0,
            "reasoning_effort": "none",
            "max_tokens": 700,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "fabjudge_review_evidence", "strict": True, "schema": output_schema},
            },
        }
        body_bytes = canonical_json(body).encode("utf-8")
        digest = hashlib.sha256(body_bytes).hexdigest()
        cache_path = self.cache_dir / f"{digest}.json"
        if cache_path.is_file():
            self.cache_hits += 1
            try:
                record = json.loads(cache_path.read_text(encoding="utf-8"))
                if record.get("request_sha256") != digest or record.get("key_role") != "llm":
                    raise ValueError("Cache identity mismatch")
                result = self._parse_response(record["response"], True, 0.0)
                self.attempts.append(self._attempt_row(digest, result))
                return result
            except Exception as exc:
                result = {"status": "MODEL_UNAVAILABLE", "error": _safe_error(exc, self._key),
                          "output": None, "cache_hit": True}
                self.attempts.append(self._attempt_row(digest, result))
                return result
        if self.new_calls >= self.max_new_calls:
            raise RuntimeError("LLM inference call cap reached")
        self.new_calls += 1
        started = time.perf_counter()
        try:
            response = self.session.post(
                OPENROUTER_CHAT_URL,
                data=body_bytes,
                headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
                timeout=HTTP_TIMEOUT_SECONDS,
            )
            if self._key in response.text:
                result = {"status": "MODEL_UNAVAILABLE", "error": "Credential appeared as [REDACTED]; refusing to cache",
                          "output": None, "cache_hit": False,
                          "latency_seconds": time.perf_counter() - started, "credential_leak": True}
                self.attempts.append(self._attempt_row(digest, result))
                return result
            if response.status_code != 200:
                result = {"status": "MODEL_UNAVAILABLE", "error": f"HTTP {response.status_code}",
                          "output": None, "cache_hit": False,
                          "latency_seconds": time.perf_counter() - started}
            else:
                raw = response.json()
                result = self._parse_response(raw, False, time.perf_counter() - started)
                if result["status"] == "success":
                    _safe_cache_write(cache_path, {
                        "serialization_version": SERIALIZATION_VERSION,
                        "request_sha256": digest,
                        "key_role": "llm", "key_env_var": LLM_KEY_ENV,
                        "model_id": LLM_MODEL_ID, "response": raw,
                        "latency_seconds": result["latency_seconds"],
                    }, self._key)
        except Exception as exc:
            result = {"status": "MODEL_UNAVAILABLE", "error": _safe_error(exc, self._key),
                      "output": None, "cache_hit": False,
                      "latency_seconds": time.perf_counter() - started}
        self.attempts.append(self._attempt_row(digest, result))
        return result

    def _parse_response(self, raw: dict, cache_hit: bool, latency: float) -> dict:
        try:
            if not isinstance(raw, dict):
                raise ValueError("LLM response is not an object")
            returned_model_id = raw.get("model")
            if not isinstance(returned_model_id, str) or not returned_model_id:
                raise ValueError("LLM response has no model ID")
            choices = raw.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError("LLM response has no choice")
            message = choices[0].get("message")
            if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                raise ValueError("LLM response has no JSON content")
            output = json.loads(message["content"])
            validate_llm_output(output)
            usage = raw.get("usage")
            return {"status": "success", "output": output, "usage": usage,
                    "returned_model_id": returned_model_id,
                    "cost_usd": _cost_from_usage(usage),
                    "latency_seconds": float(latency), "cache_hit": cache_hit, "error": None}
        except Exception as exc:
            return {"status": "MODEL_UNAVAILABLE", "output": None,
                    "usage": raw.get("usage") if isinstance(raw, dict) else None,
                    "returned_model_id": raw.get("model") if isinstance(raw, dict) else None,
                    "cost_usd": _cost_from_usage(raw.get("usage") if isinstance(raw, dict) else None),
                    "latency_seconds": float(latency), "cache_hit": cache_hit,
                    "error": _safe_error(exc, self._key)}

    @staticmethod
    def _attempt_row(digest: str, result: dict) -> dict:
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        return {"request_sha256": digest, "key_role": "llm", "key_env_var": LLM_KEY_ENV,
                "model_id": LLM_MODEL_ID, "status": result["status"],
                "cache_hit": bool(result.get("cache_hit")),
                "latency_seconds": result.get("latency_seconds"),
                "cost_usd": result.get("cost_usd"),
                "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
                "output_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
                "error": result.get("error")}
