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

from config import PipelineConfig, PipelineMode, SelectionPolicy
from run_store import RunStore

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing. Set it in your .env or environment.")


# -------------------------
# Logging (console only)
# -------------------------

class ContextFilter(logging.Filter):
    """Ensures every log record has run_id and step, so formatting never crashes."""
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
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

root_logger = logging.getLogger()
for h in root_logger.handlers:
    h.addFilter(ContextFilter())


class StepLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return msg, {"extra": {**self.extra, **kwargs.get("extra", {})}}


# -------------------------
# Models
# -------------------------

class AgentStep(str, Enum):
    COLLECT_REQUIREMENTS = "collect_requirements"
    ANALYZE_REQUIREMENTS = "analyze_requirements"
    PROPOSE_DIRECTIONS = "propose_directions"
    GENERATE_DRAFT = "generate_draft"
    REFINE_OUTPUT = "refine_output"
    FORMAT_FINAL_OUTPUT = "format_final_output"


class Requirements(BaseModel):
    task: str = Field(..., min_length=1)


class RequirementsAnalysis(BaseModel):
    missing_info: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)


class DirectionOption(BaseModel):
    name: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)


class DirectionsProposal(BaseModel):
    options: List[DirectionOption] = Field(..., min_length=1)


class DraftPrompt(BaseModel):
    persona: str = Field(..., min_length=1)
    context: str = Field(..., min_length=1)
    task: str = Field(..., min_length=1)
    output_requirements: str = Field(..., min_length=1)
    permission_to_fail: str = Field(..., min_length=1)


class RefinementReport(BaseModel):
    changes: List[str] = Field(default_factory=list)
    refined: DraftPrompt


class FinalPrompt(BaseModel):
    prompt: str = Field(..., min_length=1)


class RunState(BaseModel):
    run_id: str = Field(..., min_length=1)
    step: AgentStep
    requirements: Optional[Requirements] = None
    analysis: Optional[RequirementsAnalysis] = None
    directions: Optional[DirectionsProposal] = None
    draft: Optional[DraftPrompt] = None
    refinement: Optional[RefinementReport] = None
    final: Optional[FinalPrompt] = None


# -------------------------
# API request/response models
# -------------------------

class PromptRequest(BaseModel):
    task: str = Field(..., min_length=1)


class PromptResponse(BaseModel):
    prompt: str


# -------------------------
# LLM client
# -------------------------

class LLMClient:
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def call(self, system_prompt: str, user_input: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=user_input,
        )
        return response.output_text


# -------------------------
# Orchestrator
# -------------------------

class PromptForgeOrchestrator:
    def __init__(self, llm: LLMClient, store: RunStore, config: PipelineConfig):
        self.llm = llm
        self.store = store
        self.config = config
        self.run_id = str(uuid.uuid4())[:8]
        self.current_task = ""
        self.log = StepLoggerAdapter(
            logging.getLogger(__name__),
            {"run_id": self.run_id, "step": "INIT"}
        )

    def _save_state(self, state: RunState) -> None:
        self.store.save(state.model_dump())

    def _call_json(self, system_prompt: str, user_input: str) -> dict:
        last_raw = ""
        retries = 2 if self.config.json_strict else 1
        for _ in range(retries):
            raw = self.llm.call(system_prompt=system_prompt, user_input=user_input)
            last_raw = raw
            try:
                return json.loads(raw)
            except Exception:
                system_prompt = (
                    system_prompt
                    + "\n\nIMPORTANT: Your last output was invalid JSON. "
                      "Return ONLY valid JSON. No prose, no code fences."
                )
        raise RuntimeError(f"Bad LLM output (invalid JSON):\n{last_raw}")

    # ✅ IMPORTANT FIX: normalize dict/list -> string for DraftPrompt fields
    def _ensure_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, indent=2)
        return str(value).strip()

    def _select_direction(self, directions: DirectionsProposal) -> DirectionOption:
        # API mode: never ask user; always pick the first option.
        return directions.options[0]

    def collect_requirements(self, user_task: str) -> Requirements:
        self.log.info("START", extra={"step": "COLLECT_REQUIREMENTS"})
        out = Requirements(task=user_task)
        self.log.info("OK", extra={"step": "COLLECT_REQUIREMENTS"})
        return out

    def analyze_requirements(self, req: Requirements) -> RequirementsAnalysis:
        self.log.info("START", extra={"step": "ANALYZE_REQUIREMENTS"})

        system_prompt = (
            "You are a requirements-analysis agent for prompt engineering.\n"
            "You MUST NOT invent user requirements.\n"
            "Identify missing info to build a high-quality prompt + minimal assumptions.\n\n"
            "Return ONLY valid JSON with keys:\n"
            "missing_info (array of strings), assumptions (array of strings).\n"
            "No extra text."
        )

        user_input = (
            f"User task:\n{req.task}\n\n"
            "Analyze what is missing to create a strong, reusable AI prompt. "
            "Missing info should be concrete (audience, tone, platform, output format). "
            "Assumptions should be explicit and minimal."
        )

        data = self._call_json(system_prompt=system_prompt, user_input=user_input)

        if not isinstance(data.get("missing_info"), list):
            data["missing_info"] = [str(data.get("missing_info"))] if data.get("missing_info") else []
        if not isinstance(data.get("assumptions"), list):
            data["assumptions"] = [str(data.get("assumptions"))] if data.get("assumptions") else []

        out = RequirementsAnalysis(**data)
        self.log.info("OK", extra={"step": "ANALYZE_REQUIREMENTS"})
        return out

    def propose_directions(self, analysis: RequirementsAnalysis) -> DirectionsProposal:
        self.log.info("START", extra={"step": "PROPOSE_DIRECTIONS"})

        system_prompt = (
            "You are a prompt-strategy orchestrator.\n"
            "Given missing info + assumptions, propose 3 distinct prompt-building directions.\n"
            "Each option must be practical and must NOT invent user requirements.\n\n"
            "Return ONLY valid JSON with this shape:\n"
            "{ \"options\": [ {\"name\": \"...\", \"rationale\": \"...\"}, ... ] }\n"
            "No extra text."
        )

        user_input = json.dumps(
            {
                "missing_info": analysis.missing_info,
                "assumptions": analysis.assumptions,
                "rule": "Do not add requirements. Only propose strategies for how to write the prompt."
            },
            ensure_ascii=False
        )

        data = self._call_json(system_prompt=system_prompt, user_input=user_input)

        opts = data.get("options", [])
        if not isinstance(opts, list):
            opts = []

        out = DirectionsProposal(**{"options": opts})
        self.log.info("OK", extra={"step": "PROPOSE_DIRECTIONS"})
        return out

    def generate_draft(self, req: Requirements, chosen: DirectionOption) -> DraftPrompt:
        self.log.info("START", extra={"step": "GENERATE_DRAFT"})

        system_prompt = (
            "You are a prompt-design agent. You NEVER execute the user's task. "
            "You only produce a prompt template.\n\n"
            "You must follow the chosen direction:\n"
            f"- Direction name: {chosen.name}\n"
            f"- Rationale: {chosen.rationale}\n\n"
            "Rules:\n"
            "- Do NOT invent user requirements.\n"
            "- Always include the foundations: persona, context, task, output_requirements, permission_to_fail.\n"
            "- Return ONLY JSON object with keys:\n"
            "persona, context, task, output_requirements, permission_to_fail\n"
            "- IMPORTANT: values must be STRINGS (not objects, not arrays).\n"
            "No extra text."
        )

        data = self._call_json(system_prompt, f"Build a prompt for: {req.task}")

        # ✅ normalize all 5 fields to strings
        data["persona"] = self._ensure_text(data.get("persona"))
        data["context"] = self._ensure_text(data.get("context"))
        data["task"] = self._ensure_text(data.get("task"))
        data["output_requirements"] = self._ensure_text(data.get("output_requirements"))
        data["permission_to_fail"] = self._ensure_text(data.get("permission_to_fail"))

        out = DraftPrompt(**data)

        self.log.info("OK", extra={"step": "GENERATE_DRAFT"})
        return out

    def refine_output(self, draft: DraftPrompt, chosen: DirectionOption) -> RefinementReport:
        self.log.info("START", extra={"step": "REFINE_OUTPUT"})

        system_prompt = (
            "You are a prompt-quality refiner. You NEVER execute the user's task. "
            "You only improve the prompt.\n\n"
            "You must preserve and strengthen the chosen direction:\n"
            f"- Direction name: {chosen.name}\n"
            f"- Rationale: {chosen.rationale}\n\n"
            "Rules:\n"
            "- Do NOT invent user requirements.\n"
            "- Do NOT change the task intent.\n"
            "- Keep foundations: persona, context, task, output_requirements, permission_to_fail.\n\n"
            "Return ONLY JSON with keys:\n"
            "changes (array of strings), refined (object with keys: "
            "persona, context, task, output_requirements, permission_to_fail).\n"
            "- IMPORTANT: all refined values must be STRINGS.\n"
            "No extra text."
        )

        user_input = json.dumps(draft.model_dump(), ensure_ascii=False)
        data = self._call_json(system_prompt=system_prompt, user_input=user_input)

        refined = data.get("refined", {})

        # ✅ normalize refined fields too
        refined["persona"] = self._ensure_text(refined.get("persona"))
        refined["context"] = self._ensure_text(refined.get("context"))
        refined["task"] = self._ensure_text(refined.get("task"))
        refined["output_requirements"] = self._ensure_text(refined.get("output_requirements"))
        refined["permission_to_fail"] = self._ensure_text(refined.get("permission_to_fail"))
        data["refined"] = refined

        out = RefinementReport(**data)

        self.log.info("OK", extra={"step": "REFINE_OUTPUT"})
        return out

    def format_final_output(self, refined: DraftPrompt) -> FinalPrompt:
        self.log.info("START", extra={"step": "FORMAT_FINAL_OUTPUT"})

        prompt = (
            f"ROLE / PERSONA:\n{refined.persona}\n\n"
            f"CONTEXT:\n{refined.context}\n\n"
            f"TASK:\n{refined.task}\n\n"
            f"OUTPUT REQUIREMENTS:\n{refined.output_requirements}\n\n"
            f"PERMISSION TO FAIL:\n{refined.permission_to_fail}"
        )
        out = FinalPrompt(prompt=prompt)

        self.log.info("OK", extra={"step": "FORMAT_FINAL_OUTPUT"})
        return out

    def run(self, user_task: str) -> FinalPrompt:
        self.current_task = user_task
        self.log.info("START", extra={"step": "RUN"})

        state = RunState(run_id=self.run_id, step=AgentStep.COLLECT_REQUIREMENTS)
        self._save_state(state)

        req = self.collect_requirements(user_task)
        state.step = AgentStep.COLLECT_REQUIREMENTS
        state.requirements = req
        self._save_state(state)

        analysis = self.analyze_requirements(req)
        state.step = AgentStep.ANALYZE_REQUIREMENTS
        state.analysis = analysis
        self._save_state(state)

        directions = self.propose_directions(analysis)
        state.step = AgentStep.PROPOSE_DIRECTIONS
        state.directions = directions
        self._save_state(state)

        chosen = self._select_direction(directions)

        # ✅ store chosen direction so history/suggestion works
        self.store.save({
            "run_id": self.run_id,
            "task": self.current_task,
            "chosen_direction_name": chosen.name,
            "step": "CHOSEN_DIRECTION",
        })

        # ✅ log chosen once
        self.log.info(f"CHOSEN: {chosen.name}", extra={"step": "PROPOSE_DIRECTIONS"})

        draft = self.generate_draft(req, chosen)
        state.step = AgentStep.GENERATE_DRAFT
        state.draft = draft
        self._save_state(state)

        refinement = self.refine_output(draft, chosen)
        refined = refinement.refined
        state.step = AgentStep.REFINE_OUTPUT
        state.refinement = refinement
        self._save_state(state)

        final = self.format_final_output(refined)
        state.step = AgentStep.FORMAT_FINAL_OUTPUT
        state.final = final
        self._save_state(state)

        self.log.info("OK", extra={"step": "RUN"})
        return final


# -------------------------
# FastAPI app
# -------------------------

app = FastAPI(title="Prompt Maker Backend", version="1.0.0")

# CORS for your Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_config = PipelineConfig.from_env()
_store = RunStore()
_llm = LLMClient(api_key=OPENAI_API_KEY, model=_config.model)

# Create orchestrator once (fine for local dev)
_orchestrator = PromptForgeOrchestrator(_llm, _store, _config)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/prompt", response_model=PromptResponse)
def make_prompt(req: PromptRequest) -> PromptResponse:
    try:
        final = _orchestrator.run(req.task)
        return PromptResponse(prompt=final.prompt)
    except Exception as e:
        # keep error readable in logs
        logging.getLogger(__name__).exception("ERROR in /api/prompt")
        raise HTTPException(status_code=500, detail=str(e))
