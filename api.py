# api.py
from __future__ import annotations

import os
import json
import uuid
import logging
from enum import Enum
from typing import List, Optional, Any, Dict

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import PipelineConfig, PipelineMode
from run_store import RunStore


# ============================================================
# ENV
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")

# ============================================================
# LOGGING
# ============================================================

class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = getattr(record, "run_id", "-")
        record.step = getattr(record, "step", "-")
        return True


logging.basicConfig(
    level=logging.INFO,
    format="[RUN %(run_id)s] STEP=%(step)s → %(message)s",
)

for h in logging.getLogger().handlers:
    h.addFilter(ContextFilter())


class StepLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return msg, {"extra": {**self.extra, **kwargs.get("extra", {})}}


# ============================================================
# MODELS
# ============================================================

class AgentStep(str, Enum):
    COLLECT_REQUIREMENTS = "collect_requirements"
    ANALYZE_REQUIREMENTS = "analyze_requirements"
    PROPOSE_DIRECTIONS = "propose_directions"
    GENERATE_DRAFT = "generate_draft"
    FORMAT_FINAL_OUTPUT = "format_final_output"


class Requirements(BaseModel):
    task: str


class RequirementsAnalysis(BaseModel):
    missing_info: List[str] = []
    assumptions: List[str] = []


class DirectionOption(BaseModel):
    name: str
    rationale: str


class DirectionsProposal(BaseModel):
    options: List[DirectionOption]


class DraftPrompt(BaseModel):
    persona: str
    context: str
    task: str
    output_requirements: str
    permission_to_fail: str


class FinalPrompt(BaseModel):
    prompt: str


class PromptRequest(BaseModel):
    task: str


class PromptResponse(BaseModel):
    prompt: str


# ============================================================
# LLM CLIENT (SAFE)
# ============================================================

class LLMClient:
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key, timeout=25)
        self.model = model

    def call(self, system_prompt: str, user_input: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=user_input,
            max_output_tokens=700,
        )
        return response.output_text


# ============================================================
# ORCHESTRATOR (FAST PATH)
# ============================================================

class PromptForgeOrchestrator:
    def __init__(self, llm: LLMClient, store: RunStore, config: PipelineConfig):
        self.llm = llm
        self.store = store
        self.config = config
        self.run_id = str(uuid.uuid4())[:8]
        self.log = StepLoggerAdapter(
            logging.getLogger(__name__),
            {"run_id": self.run_id, "step": "INIT"},
        )

    def _call_json(self, system_prompt: str, user_input: str) -> dict:
        raw = self.llm.call(system_prompt, user_input)
        try:
            return json.loads(raw)
        except Exception:
            raise RuntimeError("Invalid JSON from LLM")

    def _ensure_text(self, v: Any) -> str:
        if isinstance(v, str):
            return v.strip()
        return json.dumps(v, ensure_ascii=False)

    def run(self, user_task: str) -> FinalPrompt:
        self.log.info("START", extra={"step": "RUN"})

        # 1. ANALYZE
        analysis = self._call_json(
            system_prompt=(
                "Analyze missing info and assumptions.\n"
                "Return JSON {missing_info:[], assumptions:[]}"
            ),
            user_input=user_task,
        )

        # 2. DIRECTIONS
        directions = self._call_json(
            system_prompt=(
                "Propose 3 prompt-building strategies.\n"
                "Return JSON {options:[{name,rationale}]}"
            ),
            user_input=json.dumps(analysis),
        )

        chosen = directions["options"][0]

        # 3. DRAFT (MAIN VALUE)
        draft = self._call_json(
            system_prompt=(
                "You design prompts only.\n"
                "Return JSON with persona, context, task, "
                "output_requirements, permission_to_fail (strings)."
            ),
            user_input=f"Task: {user_task}\nDirection: {chosen['name']}",
        )

        draft = DraftPrompt(
            persona=self._ensure_text(draft["persona"]),
            context=self._ensure_text(draft["context"]),
            task=self._ensure_text(draft["task"]),
            output_requirements=self._ensure_text(draft["output_requirements"]),
            permission_to_fail=self._ensure_text(draft["permission_to_fail"]),
        )

        prompt = (
            f"ROLE / PERSONA:\n{draft.persona}\n\n"
            f"CONTEXT:\n{draft.context}\n\n"
            f"TASK:\n{draft.task}\n\n"
            f"OUTPUT REQUIREMENTS:\n{draft.output_requirements}\n\n"
            f"PERMISSION TO FAIL:\n{draft.permission_to_fail}"
        )

        self.log.info("OK", extra={"step": "RUN"})
        return FinalPrompt(prompt=prompt)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="Prompt Maker Backend", version="1.0.0")

allow_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

frontend_origin = os.getenv("FRONTEND_ORIGIN")
if frontend_origin:
    allow_origins.append(frontend_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

_config = PipelineConfig.from_env()
_store = RunStore()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/prompt", response_model=PromptResponse)
def make_prompt(req: PromptRequest):
    try:
        llm = LLMClient(
            api_key=OPENAI_API_KEY,
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        )
        orchestrator = PromptForgeOrchestrator(llm, _store, _config)
        final = orchestrator.run(req.task)
        return PromptResponse(prompt=final.prompt)

    except Exception as e:
        logging.exception("API ERROR")
        raise HTTPException(status_code=500, detail=str(e))
