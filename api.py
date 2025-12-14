# api.py
from __future__ import annotations

import os
import json
import uuid
import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import PipelineConfig
from run_store import RunStore

# Load .env locally (Render uses env vars)
load_dotenv()

# -------------------------
# Logging
# -------------------------

class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id"):
            record.run_id = "-"
        if not hasattr(record, "step"):
            record.step = "-"
        return True

logging.basicConfig(
    level=logging.INFO,
    format="[RUN %(run_id)s] STEP=%(step)s → %(message)s"
)
root_logger = logging.getLogger()
for h in root_logger.handlers:
    h.addFilter(ContextFilter())

class StepLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return msg, {"extra": {**self.extra, **kwargs.get("extra", {})}}

# -------------------------
# Models
# -------------------------

class PromptRequest(BaseModel):
    task: str = Field(..., min_length=1)

class PromptResponse(BaseModel):
    prompt: str

class DraftPrompt(BaseModel):
    persona: str = Field(..., min_length=1)
    context: str = Field(..., min_length=1)
    task: str = Field(..., min_length=1)
    output_requirements: str = Field(..., min_length=1)
    permission_to_fail: str = Field(..., min_length=1)

# -------------------------
# Helpers
# -------------------------

def ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value).strip()

def call_json(client: OpenAI, model: str, system_prompt: str, user_input: str, retries: int = 2) -> dict:
    last_raw = ""
    for _ in range(retries):
        resp = client.responses.create(
            model=model,
            instructions=system_prompt,
            input=user_input,
        )
        raw = (resp.output_text or "").strip()
        last_raw = raw
        if not raw:
            continue
        try:
            return json.loads(raw)
        except Exception:
            system_prompt += (
                "\n\nIMPORTANT: Your last output was invalid JSON. "
                "Return ONLY valid JSON. No prose, no code fences."
            )
    raise RuntimeError(f"Invalid JSON from LLM. Raw output:\n{last_raw}")

def format_final(d: DraftPrompt) -> str:
    return (
        f"ROLE / PERSONA:\n{d.persona}\n\n"
        f"CONTEXT:\n{d.context}\n\n"
        f"TASK:\n{d.task}\n\n"
        f"OUTPUT REQUIREMENTS:\n{d.output_requirements}\n\n"
        f"PERMISSION TO FAIL:\n{d.permission_to_fail}"
    )

# -------------------------
# FastAPI app
# -------------------------

app = FastAPI(title="Prompt Maker Backend", version="1.0.0")

frontend_origin = os.getenv("FRONTEND_ORIGIN", "").strip()
allow_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
if frontend_origin:
    allow_origins.append(frontend_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_config = PipelineConfig.from_env()
_store = RunStore()

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/api/ping")
def ping() -> Dict[str, str]:
    return {"ok": "pong"}

@app.post("/api/prompt")
def make_prompt(req: PromptRequest):
    run_id = str(uuid.uuid4())[:8]
    log = StepLoggerAdapter(logging.getLogger(__name__), {"run_id": run_id, "step": "API"})

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is missing on the server.")

    # BIGGER timeout so Render free doesn’t die mid-call
    # This is the core fix for your ReadTimeout.
    client = OpenAI(api_key=api_key, timeout=90.0)

    model = (_config.model or "gpt-5").strip()

    log.info("START", extra={"step": "PROMPT"})

    system_prompt = (
        "You are a prompt-design agent.\n"
        "You NEVER execute the user's task.\n"
        "You ONLY output a reusable prompt.\n\n"
        "Return ONLY valid JSON with EXACT keys:\n"
        "persona, context, task, output_requirements, permission_to_fail\n"
        "All values MUST be strings.\n"
        "No extra text."
    )

    user_input = f"Build a high-quality prompt for this user task:\n{req.task}"

    try:
        data = call_json(client, model, system_prompt, user_input, retries=2)

        # normalize
        data["persona"] = ensure_text(data.get("persona"))
        data["context"] = ensure_text(data.get("context"))
        data["task"] = ensure_text(data.get("task"))
        data["output_requirements"] = ensure_text(data.get("output_requirements"))
        data["permission_to_fail"] = ensure_text(data.get("permission_to_fail"))

        draft = DraftPrompt(**data)
        final_prompt = format_final(draft)

        # log/store
        _store.save({"run_id": run_id, "step": "DONE", "task": req.task})

        log.info("OK", extra={"step": "PROMPT"})

        return JSONResponse(content={"prompt": final_prompt})

    except Exception as e:
        log.exception("ERROR", extra={"step": "PROMPT"})
        raise HTTPException(status_code=500, detail=str(e))
