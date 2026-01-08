# promptforge_core.py
import os
import json
import uuid
import logging
from enum import Enum
from typing import List, Optional, Union, Any
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, model_validator


from config import PipelineConfig, PipelineMode, SelectionPolicy
from run_store import RunStore


load_dotenv()


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
    persona: str = Field(default="", min_length=0)
    context: str = Field(default="", min_length=0)
    task: str = Field(..., min_length=1)
    output_requirements: str = Field(default="", min_length=0)
    permission_to_fail: Union[str, bool] = Field(default="")

    @model_validator(mode="before")
    @classmethod
    def ensure_strings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
            
        def stringify(val: Any) -> str:
            if isinstance(val, (dict, list)):
                return json.dumps(val, ensure_ascii=False)
            return str(val)

        for field in ["persona", "context", "output_requirements", "task"]:
            if field in data and not isinstance(data[field], str):
                data[field] = stringify(data[field])
        
        return data



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
# LLM client
# -------------------------

class LLMClient:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
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
        self.log = StepLoggerAdapter(
            logging.getLogger(__name__),
            {"run_id": self.run_id, "step": "INIT"}
        )
        self.current_task = ""

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

    def _select_direction(self, directions: DirectionsProposal) -> DirectionOption:
        """
        IMPORTANT:
        - In API/production flow we do NOT ask user to choose.
        - Even if running in dev mode, SelectionPolicy can be forced to 'first'.
        """
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
            "Your job: identify what info is missing to build a high-quality prompt, "
            "and list any assumptions you would otherwise be forced to make.\n\n"
            "IMPORTANT: If the user mission includes 'FINAL GENERATION DIRECTIVE' or "
            "'USER PROVIDED ANSWERS', do NOT look for tiny missing details. "
            "If the info is sufficient to build a good prompt, return EMPTY arrays.\n\n"
            "Return ONLY valid JSON with keys:\n"
            "missing_info (array of strings), assumptions (array of strings).\n"
            "No extra text."
        )

        user_input = (
            f"User task:\n{req.task}\n\n"
            "Analyze what is missing to create a strong, reusable AI prompt. "
            "Missing info should be concrete (e.g., audience, tone, platform, output format). "
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
            "IMPORTANT: If the user provided detailed answers, focus your options on "
            "EXPLOITING those details to build the best possible prompt, "
            "NOT on asking for more information.\n\n"
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
            "You are a MASTER PROMPT ARCHITECT using the 4W Framework to build production-grade AI prompts.\n\n"
            "Follow the chosen direction:\n"
            f"- Strategy: {chosen.name}\n"
            f"- Rationale: {chosen.rationale}\n\n"
            "REQUIRED 4W STRUCTURE - Build the prompt with these sections:\n\n"
            "1. ROLE (Who): Define a specific persona with domain expertise relevant to the task. "
            "Be sharp and data-oriented. Example: 'Act as a sharp, data-oriented private-equity analyst...'\n\n"
            "2. OBJECTIVE (Why): State the clear goal, what success looks like, and why this matters. "
            "Be specific about deliverables (e.g., 'Help me draft a single-page benchmark summary...').\n\n"
            "3. CONTEXT PACKAGE (What): Include ALL of these sub-elements:\n"
            "   - Audience: Who will consume the output\n"
            "   - Voice & Tone: Formal/casual, insight-driven, no fluff, etc.\n"
            "   - Length Target: Specific word count or page length\n"
            "   - Key Facts/Data: What information MUST be used\n"
            "   - Known Constraints: Citations required, sources to stay within, things to avoid\n\n"
            "4. WORKFLOW (How): Define a step-by-step process:\n"
            "   - Step 0: GAP CHECK - List any missing information; ask concise questions until gaps are filled\n"
            "   - Step 1: PLAN - Outline a logical structure; wait for approval\n"
            "   - Step 2: DRAFT - Write first version following approved plan\n"
            "   - Step 3: REVIEW - Pause and ask for feedback on clarity, tone, completeness\n"
            "   - Step 4: REVISE - Improve with notes; repeat until user agrees\n\n"
            "5. CONTEXT-HANDLING RULES: How to handle edge cases:\n"
            "   - If input exceeds ~200 words, summarize first and ask whether to keep full text\n"
            "   - If external knowledge is needed, list it in Gap Check instead of guessing\n\n"
            "6. FIRST ACTION: Always instruct to begin with 'Gap Check' step.\n\n"
            "STRICT RULES:\n"
            "- Every word must add functional value (NO FILLER)\n"
            "- Expand every user detail into specific instructions\n"
            "- Use sophisticated prompt engineering language\n\n"
            "Return ONLY a JSON object with these keys (each mapping to a rich, long STRING):\n"
            "persona, context, task, output_requirements, permission_to_fail\n"
            "No extra text."
        )

        data = self._call_json(system_prompt, f"Build a prompt for: {req.task}")

        if isinstance(data.get("output_requirements"), list):
            data["output_requirements"] = "\n".join(map(str, data["output_requirements"]))

        out = DraftPrompt(**data)

        self.log.info("OK", extra={"step": "GENERATE_DRAFT"})
        return out

    def refine_output(self, draft: DraftPrompt, chosen: DirectionOption) -> RefinementReport:
        self.log.info("START", extra={"step": "REFINE_OUTPUT"})

        system_prompt = (
            "You are a SENIOR PROMPT OPTIMIZER specializing in the 4W Framework.\n"
            "Your job is to validate and enhance the draft prompt to be 2x more detailed and robust.\n\n"
            "4W FRAMEWORK VALIDATION - Ensure the prompt has ALL of these:\n"
            "1. ROLE (Who): Clear persona with domain expertise - strengthen with specific credentials\n"
            "2. OBJECTIVE (Why): Explicit goal with success criteria - add measurable outcomes\n"
            "3. CONTEXT PACKAGE (What): Complete with Audience, Voice/Tone, Length, Key Facts, Constraints\n"
            "4. WORKFLOW (How): Step-by-step process with 'Gap Check' as Step 0\n"
            "5. CONTEXT-HANDLING RULES: How to handle long inputs or missing knowledge\n"
            "6. FIRST ACTION: Directive to begin with Gap Check\n\n"
            "REFINEMENT DIRECTIVES:\n"
            "- If any 4W section is weak or missing, EXPAND it significantly\n"
            "- Strengthen the persona with domain-specific expertise and credentials\n"
            "- Add 'Success Criteria' and 'Negative Constraints' in output_requirements if missing\n"
            "- Ensure the prompt is 'leak-proof' and handles edge cases elegantly\n"
            "- WORKFLOW must always include Gap Check as the first step\n"
            f"- Maintain the chosen direction: {chosen.name}\n\n"
            "Return ONLY JSON with keys:\n"
            "changes (array of strings), refined (object with keys: "
            "persona, context, task, output_requirements, permission_to_fail).\n"
            "Every field in 'refined' MUST be a high-density string."
        )

        user_input = json.dumps(draft.model_dump(), ensure_ascii=False)
        data = self._call_json(system_prompt=system_prompt, user_input=user_input)

        refined = data.get("refined", {})
        if isinstance(refined.get("output_requirements"), list):
            refined["output_requirements"] = "\n".join(map(str, refined["output_requirements"]))
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

    def run(self, user_task: str, interaction_callback: Optional[callable] = None, answers: Optional[dict] = None) -> FinalPrompt:
        self.current_task = user_task
        self.log.info("START", extra={"step": "RUN"})

        # Enrich task with answers immediately if provided
        if answers:
            self.current_task += "\n\n[USER ANSWERS TO CLARIFYING QUESTIONS]:\n"
            for q, a in answers.items():
                self.current_task += f"Q: {q}\nA: {a}\n"

        state = RunState(run_id=self.run_id, step=AgentStep.COLLECT_REQUIREMENTS)
        self._save_state(state)

        req = self.collect_requirements(self.current_task)
        state.step = AgentStep.COLLECT_REQUIREMENTS
        state.requirements = req
        self._save_state(state)

        analysis = self.analyze_requirements(req)
        state.step = AgentStep.ANALYZE_REQUIREMENTS
        state.analysis = analysis
        self._save_state(state)

        # INTERACTIVE CHECK
        # If we have missing info, we check if we can get it from callback
        # (If answers were already passed, they are in the task now, so analysis should be cleaner.
        # But if analysis STILL finds missing info, we ask again.)
        
        if analysis.missing_info:
            additional_info = None
            if interaction_callback:
                self.log.info("INTERACTIVE: Asking user for input via callback", extra={"step": "ANALYZE_REQUIREMENTS"})
                additional_info = interaction_callback(analysis.missing_info)
            
            # If we got info (either from callback or future logic), update task
            if additional_info:
                 req.task += f"\n\n[USER PROVIDED DETAILS]:\n{additional_info}"
                 # We could re-analyze here, but we proceed for now
                 state.requirements = req
                 self._save_state(state)

        directions = self.propose_directions(analysis)
        state.step = AgentStep.PROPOSE_DIRECTIONS
        state.directions = directions
        self._save_state(state)

        chosen = self._select_direction(directions)

        # Store a minimal record so suggestion can work later (if you ever enable it)
        self.store.save({
            "run_id": self.run_id,
            "task": self.current_task,
            "chosen_direction_name": chosen.name,
            "step": "CHOSEN_DIRECTION",
        })

        # Log once
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


def build_orchestrator() -> PromptForgeOrchestrator:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = "MISSING" # Allow building for inspection/logs, but call() will fail if used.

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


    config = PipelineConfig.from_env()

    # IMPORTANT: API flow should never ask user to choose:
    config.mode = PipelineMode.prod
    config.selection_policy = SelectionPolicy.first

    llm = LLMClient(api_key=api_key, model=model)
    store = RunStore()
    return PromptForgeOrchestrator(llm, store, config=config)
