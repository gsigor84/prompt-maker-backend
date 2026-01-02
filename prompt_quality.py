from __future__ import annotations

import json
import logging
import os
from pydantic import BaseModel, Field
from openai import OpenAI

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger(__name__)

# -------------------------
# Models
# -------------------------

class QualityScore(BaseModel):
    score: int = Field(..., description="Score from 0 to 10")
    critique: str = Field(..., description="Short explanation of the score and suggestions for improvement.")

# -------------------------
# Service
# -------------------------

class PromptQualityService:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing")
            
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def score_prompt(self, prompt_text: str, original_task: str) -> QualityScore:
        """
        Evaluates the given prompt against the original user task.
        Returns a QualityScore (0-10) and critique.
        """
        system_prompt = (
            "You are a strict prompt engineering critic.\n"
            "Evaluate the provided AI system prompt based on how well it solves the user's original task.\n\n"
            "Criteria:\n"
            "1. Clarity: Is the role and context clear?\n"
            "2. Structure: Does it have distinct sections (Role, Context, Task, etc.)?\n"
            "3. Robustness: Does it handle edge cases or set constraints?\n"
            "4. Completeness: Does it fully address the original user task?\n\n"
            "Return ONLY valid JSON with keys:\n"
            "\"score\" (integer 0-10), \"critique\" (string, max 50 words)."
        )

        user_input = (
            f"ORIGINAL USER TASK:\n{original_task}\n\n"
            f"GENERATED PROMPT TO EVALUATE:\n{prompt_text}"
        )

        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=user_input
            )
            raw = response.output_text
            # Basic cleanup/parsing
            data = json.loads(raw)
            return QualityScore(**data)
        except Exception as e:
            logger.error(f"Failed to score prompt: {e}")
            # Fallback
            return QualityScore(score=0, critique=f"Scoring failed: {str(e)}")
