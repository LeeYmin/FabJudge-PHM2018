"""Fixed identifiers and safe defaults for the 07a development run."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "07a_pipeline_skeleton"

JEV_KEY_ENV = "JEV_API_KEY"
LLM_KEY_ENV = "GPT_LUNA_API_KEY"
JEV_MODEL_ID = "typesafe/jev-1.13"
LLM_MODEL_ID = "~openai/gpt-luna-latest"
JEV_URL = "https://openrouter.ai/api/alpha/decisions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

PACKET_VERSION = "07a-evidence-v1"
SERIALIZATION_VERSION = "v2"
RANDOM_SEED = 20260927
SAMPLE_CAP = 600
LOW_AUDIT_RATE = 0.10
MAX_JEV_EXTRA_CALLS = 15
LLM_SMOKE_SIZE = 10
LLM_MAX_OUTPUT_TOKENS = 700
HTTP_TIMEOUT_SECONDS = 90

