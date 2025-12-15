# api.py
from __future__ import annotations

import os
import json
import uuid
import time
import logging
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI
from openai import APITimeoutError, APIConnectionError, RateLimitError, APIError
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


def extract_json(raw: str) -> dict:
    """
    Best-effort JSON extraction:
    - If raw is valid JSON -> return it
    - Else try to extract the first {...} block
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Empty output")

    # Direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Extract first JSON object block
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start:end + 1].strip()
        return json.loads(candidate)

    raise ValueError("No JSON object found")


def call_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_input: str,
    *,
    attempts: int = 3,
) -> dict:
    """
    Render free tier can be slow/spin down, so:
    - retry on transient OpenAI/http issues
    - retry on invalid JSON
    """
    last_raw = ""
    prompt = system_prompt

    for i in range(1, attempts + 1):
        try:
            resp = client.responses.create(
                model=model,
                instructions=prompt,
                input=user_input,
            )

            raw = (resp.output_text or "").strip()
            last_raw = raw

            data = extract_json(raw)

            # Normalize required keys so DraftPrompt doesn't crash
            for k in ["persona", "context", "task", "output_requirements", "permission_to_fail"]:
                data[k] = ensure_text(data.get(k))

            return data

        except (APITimeoutError, APIConnectionError, RateLimitError, APIError) as e:
            # Backoff on API/network transient issues
            wait = min(2 ** (i - 1), 8)
            time.sleep(wait)
            if i == attempts:
                raise RuntimeError(f"OpenAI request failed after retries: {type(e).__name__}: {e}") from e

        except (json.JSONDecodeError, ValueError) as e:
            # Harden the instruction and retry
            prompt = (
                system_prompt
                + "\n\nIMPORTANT:\n"
                  "- Return ONLY a JSON object.\n"
                  "- No extra text.\n"
                  "- No markdown.\n"
                  "- Must include keys: persona, context, task, output_requirements, permission_to_fail.\n"
                  "- All values must be strings.\n"
            )
            if i == attempts:
                snippet = (last_raw[:800] + "…") if len(last_raw) > 800 else last_raw
                raise RuntimeError(
                    "Invalid JSON from LLM after retries. Last output snippet:\n"
                    + snippet
                ) from e

    raise RuntimeError("Failed to get valid JSON (unexpected).")


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

    # IMPORTANT: set client timeout here (more reliable than per-request timeout)
    client = OpenAI(api_key=api_key, timeout=90.0)

    model = (_config.model or "gpt-4o-mini").strip()

    log.info("START", extra={"step": "PROMPT"})

    system_prompt = (
        "You are a senior SQL/Database Performance Engineer and prompt designer.\n"
        "You NEVER execute the user's task.\n"
        "You ONLY output a reusable prompt for another AI to execute.\n\n"
        "Return ONLY valid JSON with EXACT keys:\n"
        "persona, context, task, output_requirements, permission_to_fail\n"
        "Rules:\n"
        "- persona MUST be exactly: \"SQL/Database Performance Expert\"\n"
        "- All values MUST be strings.\n"
        "- No extra text, no markdown, no code fences."
    )

    user_input = f"Build a high-quality prompt for this user task:\n{req.task}"

    try:
        data = call_json(
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_input=user_input,
            attempts=3,
        )

        draft = DraftPrompt(**data)
        final_prompt = format_final(draft)

        # Store should never break the response
        try:
            _store.save({"run_id": run_id, "step": "DONE", "task": req.task})
        except Exception:
            log.exception("RunStore save failed", extra={"step": "STORE"})

        log.info("OK", extra={"step": "PROMPT"})
        return JSONResponse(content={"prompt": final_prompt})

    except Exception as e:
        log.exception("ERROR", extra={"step": "PROMPT"})
        raise HTTPException(status_code=500, detail=str(e))
