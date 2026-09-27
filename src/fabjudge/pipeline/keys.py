"""Strict role-specific OpenRouter credential loading for the 07 pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from .config import JEV_KEY_ENV, LLM_KEY_ENV, PROJECT_ROOT


class CredentialConfigurationError(RuntimeError):
    """A required role-specific credential is absent or misconfigured."""


def _dotenv_values(root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    env_path = Path(root) / ".env"
    if not env_path.is_file():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name in {JEV_KEY_ENV, LLM_KEY_ENV}:
            values[name] = value
    return values


def _effective_values(root: Path = PROJECT_ROOT) -> dict[str, str | None]:
    file_values = _dotenv_values(root)
    return {
        name: (os.environ.get(name) or file_values.get(name) or None)
        for name in (JEV_KEY_ENV, LLM_KEY_ENV)
    }


def ensure_distinct_keys(root: Path = PROJECT_ROOT) -> None:
    """Refuse shared JEV/LLM credentials without ever revealing either value."""
    values = _effective_values(root)
    jev, llm = values[JEV_KEY_ENV], values[LLM_KEY_ENV]
    if jev and llm and jev == llm:
        raise CredentialConfigurationError(
            f"{JEV_KEY_ENV} and {LLM_KEY_ENV} must contain different credentials"
        )


def load_jev_key(root: Path = PROJECT_ROOT) -> str:
    ensure_distinct_keys(root)
    value = _effective_values(root)[JEV_KEY_ENV]
    if not value:
        raise CredentialConfigurationError(f"Missing required credential: {JEV_KEY_ENV}")
    return value


def load_llm_key(root: Path = PROJECT_ROOT) -> str:
    ensure_distinct_keys(root)
    value = _effective_values(root)[LLM_KEY_ENV]
    if not value:
        raise CredentialConfigurationError(f"Missing required credential: {LLM_KEY_ENV}")
    return value
