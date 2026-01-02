from __future__ import annotations

import os
import json
import logging
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Prompt Maker Backend", version="1.0.0")

# ------------------------------------------------------------------
# ✅ CORS FIX (THIS IS THE IMPORTANT PART)
# ------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"^https://prompt-maker-.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------

class PromptRequest(BaseModel):
    task: str = Field(..., min_length=1)

# ------------------------------------------------------------------
# OpenAI client
# ------------------------------------------------------------------

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is missing")

client = OpenAI(api_key=api_key, timeout=90.0)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

# ------------------------------------------------------------------
# FULL PROMPT ENDPOINT (only returns what you want)
# ------------------------------------------------------------------

@app.post("/api/prompt/full")
def generate_full_prompt(req: PromptRequest):
    try:
        system_prompt = (
            "You are an expert prompt architect.\n"
            "Your job is to produce a COMPLETE, HIGH-QUALITY PROMPT for another AI.\n\n"
            "Rules:\n"
            "- Do NOT execute the task.\n"
            "- Do NOT summarize.\n"
            "- Produce a rich, structured prompt.\n"
            "- Use the following sections exactly:\n"
            "  ROLE / PERSONA\n"
            "  CONTEXT\n"
            "  TASK\n"
            "  OUTPUT REQUIREMENTS\n"
            "  PERMISSION TO FAIL\n\n"
            "Each section MUST contain detailed JSON blocks.\n"
            "The output MUST be long, explicit, and production-grade.\n"
            "No markdown fences. No commentary. Only the prompt."
        )

        user_input = f"Build a comprehensive AI prompt for this task:\n{req.task}"

        response = client.responses.create(
            model=MODEL,
            instructions=system_prompt,
            input=user_input,
        )

        prompt_text = (response.output_text or "").strip()
        if not prompt_text:
            raise RuntimeError("Empty response from model")

        return JSONResponse(content={"prompt": prompt_text})

    except Exception as e:
        logger.exception("Prompt generation failed")
        raise HTTPException(status_code=500, detail=str(e))

