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

# ✅ NUCLEAR CORS (Added at the very top of app lifecycle)
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# CORS is now handled in main.py to cover all routers.


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
    logger.warning("⚠️ OPENAI_API_KEY is missing. Prompt generation will fail at runtime, but server will remain up.")

# Initialize client only if key is present to avoid OpenAI library initialization errors
client = None
if api_key:
    client = OpenAI(api_key=api_key, timeout=90.0)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ------------------------------------------------------------------
@app.get("/")
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "message": "Prompt Master API is Live"}


# ------------------------------------------------------------------
# FULL PROMPT ENDPOINT (only returns what you want)
# ------------------------------------------------------------------

@app.post("/api/prompt/full")
def generate_full_prompt(req: PromptRequest):
    try:
        if not client:
            raise HTTPException(status_code=500, detail="OpenAI Client not initialized (missing API Key)")

        system_prompt = (
            "You are an expert prompt architect using the 4W Framework.\n"
            "Your job is to produce a COMPLETE, HIGH-QUALITY PROMPT for another AI.\n\n"
            "Rules:\n"
            "- Do NOT execute the task.\n"
            "- Do NOT summarize.\n"
            "- Produce a rich, structured prompt using the 4W Framework.\n\n"
            "REQUIRED STRUCTURE (use these exact section headers):\n"
            "  1. ROLE (Who) - Define a specific persona with domain expertise relevant to the task\n"
            "  2. OBJECTIVE (Why) - State the clear goal, success criteria, and why this matters\n"
            "  3. CONTEXT PACKAGE (What) - Include:\n"
            "     * Audience: Who will consume the output\n"
            "     * Voice & Tone: Formal/casual, insight-driven, no fluff\n"
            "     * Length Target: Specific word count or page length\n"
            "     * Key Facts/Data: What information MUST be used\n"
            "     * Known Constraints: Citations, sources, things to avoid\n"
            "  4. WORKFLOW (How) - Step-by-step process:\n"
            "     * Step 0: GAP CHECK - List missing info; ask questions until gaps filled\n"
            "     * Step 1: PLAN - Outline structure; wait for approval\n"
            "     * Step 2: DRAFT - Write first version\n"
            "     * Step 3: REVIEW - Ask for feedback\n"
            "     * Step 4: REVISE - Improve until user agrees\n"
            "  5. CONTEXT-HANDLING RULES - How to handle long inputs or missing knowledge\n"
            "  6. FIRST ACTION - Instruct to begin with Gap Check\n\n"
            "Each section MUST be detailed and production-grade.\n"
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

