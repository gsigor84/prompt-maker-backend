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


@dataclass
class PipelineConfig:
    mode: PipelineMode = PipelineMode.prod
    selection_policy: SelectionPolicy = SelectionPolicy.first
    json_strict: bool = True
    # Use env var OPENAI_MODEL in production; this is just a fallback.
    model: str = "gpt-5"

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        mode = os.getenv("PIPELINE_MODE", "prod").strip().lower()
        json_strict = os.getenv("JSON_STRICT", "1").strip().lower() not in {"0", "false", "no"}

        model = os.getenv("OPENAI_MODEL", cls.model).strip()

        try:
            parsed_mode = PipelineMode(mode)
        except Exception:
            parsed_mode = PipelineMode.prod

        return cls(
            mode=parsed_mode,
            selection_policy=SelectionPolicy.first,
            json_strict=json_strict,
            model=model,
        )
