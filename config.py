# config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class PipelineMode(str, Enum):
    dev = "dev"
    prod = "prod"


class SelectionPolicy(str, Enum):
    # Backend/API will NOT ask the user anything.
    # "first" means we always pick option 0.
    first = "first"


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v.strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v.strip())
    except Exception:
        return default


@dataclass
class PipelineConfig:
    # Mode
    mode: PipelineMode = PipelineMode.prod
    selection_policy: SelectionPolicy = SelectionPolicy.first

    # LLM behavior
    model: str = "gpt-4o-mini"
    json_strict: bool = True

    # Reliability
    openai_timeout_s: float = 90.0
    attempts: int = 3
    max_backoff_s: int = 8

    # Output control (task-agnostic)
    # If you want the model to choose persona per task, leave this empty via env DEFAULT_PERSONA=""
    default_persona: str = ""
    enforce_permission_to_fail_false: bool = True

    # “Full” endpoint tuning
    full_missing_info_depth: str = "deep"  # "basic" | "deep"

    # ✅ NEW: enforce longer prompts in /api/prompt/full
    min_prompt_chars: int = 2500          # bump to 6000-9000 if you want “very long”
    target_prompt_chars: int = 5000       # expansion target when too short

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        mode_raw = os.getenv("PIPELINE_MODE", "prod").strip().lower()
        try:
            mode = PipelineMode(mode_raw)
        except Exception:
            mode = PipelineMode.prod

        model = os.getenv("OPENAI_MODEL", cls.model).strip()

        default_persona = os.getenv("DEFAULT_PERSONA", "").strip()

        return cls(
            mode=mode,
            selection_policy=SelectionPolicy.first,
            json_strict=_env_bool("JSON_STRICT", True),
            model=model,
            openai_timeout_s=_env_float("OPENAI_TIMEOUT_S", 90.0),
            attempts=_env_int("OPENAI_ATTEMPTS", 3),
            max_backoff_s=_env_int("OPENAI_MAX_BACKOFF_S", 8),
            default_persona=default_persona,  # empty = let model decide from task
            enforce_permission_to_fail_false=_env_bool("ENFORCE_PERMISSION_TO_FAIL_FALSE", True),
            full_missing_info_depth=os.getenv("FULL_MISSING_INFO_DEPTH", "deep").strip().lower() or "deep",
            min_prompt_chars=_env_int("MIN_PROMPT_CHARS", 2500),
            target_prompt_chars=_env_int("TARGET_PROMPT_CHARS", 5000),
        )
