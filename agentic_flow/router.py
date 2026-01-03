from fastapi import APIRouter
from .schemas import AgentRequest, AgentResponse, PipelineStep
from promptforge_core import build_orchestrator
import re

router = APIRouter(prefix="/api/v2/agent", tags=["Agentic Pipeline"])

def parse_questions_from_text(text: str) -> list[str]:
    """Helper to extract questions from LLM output."""
    # Look for lines starting with - or 1. or *
    questions = []
    for line in text.split('\n'):
        clean = line.strip()
        # Simple heuristic: starts with number/bullet and ends with question mark
        if (clean.startswith('-') or clean.startswith('*') or clean[0:1].isdigit()) and '?' in clean:
            # Remove the bullet
            q_text = re.sub(r'^[\d\.\-\*\s]+', '', clean).strip()
            questions.append(q_text)
    
    # Fallback: if no list found, just take non-empty lines with ?
    if not questions:
        questions = [l.strip() for l in text.split('\n') if '?' in l.strip()]
        
    return questions[:15] # Cap at 15

@router.post("/run", response_model=AgentResponse)
async def run_pipeline(request: AgentRequest):
    orchestrator = build_orchestrator()
    
    # === PATH A: INTERACTIVE MODE (Round 1) ===
    # User wants interaction, and we haven't asked yet (no answers).
    if request.interactive_mode and not request.answers:
        print("🤔 Interactive Mode: Forcing Question Generation Step (Direct LLM).")
        
        # 1. Direct LLM Call to get questions (Bypassing the Orchestrator Pipeline)
        system_prompt = (
            "You are an expert AI prompt consultant. "
            "Your goal is to gather all necessary requirements to build the perfect prompt."
        )
        
        user_prompt = (
            f"User Task: '{request.task}'\n\n"
            "Analyze this request deeply. It may be vague.\n"
            "List 10 to 15 specific, essential clarifying questions I must ask the user to understand their goal perfectly.\n"
            "Consider: Persona, Audience, Tone, Context, Constraints, Output Format, and Examples.\n"
            "Return ONLY the questions as a bulleted list. No conversational filler."
        )
        
        # We use the raw LLM client from the orchestrator
        raw_text = orchestrator.llm.call(system_prompt, user_prompt)
        
        # 2. Extract questions
        questions = parse_questions_from_text(raw_text)
        
        if not questions:
            # Fallback
            questions = ["Could you provide more specific details?", "What is the target audience?", "What is the desired tone?"]

        return AgentResponse(
            status="needs_info",
            questions=questions,
            execution_trace=[
                PipelineStep(step_name="ModeManager", status="completed", output="Interactive Mode: Generated Clarifying Questions")
            ]
        )

    # === PATH B: FAST MODE / RESUME (Round 2) ===
    # Either not interactive, OR we already have answers.
    
    # 1. Prepare Context with Answers
    effective_task = request.task
    if request.answers:
        formatted_answers = "\n".join([f"Q: {k}\nA: {v}" for k, v in request.answers.items()])
        
        # We wrap the user request in a "High-Density Final Generation Directive" 
        # to ensure the LLM utilizes every bit of information.
        directive = (
            "--- HIGH-DENSITY FINAL GENERATION DIRECTIVE ---\n"
            "The user has completed a detailed interactive interview. You MUST now build the FINAL PROMPT.\n"
            "CRITICAL: Do NOT just summarize. Every User Answer below is a HARD ANCHOR POINT. "
            "You must weave these details into a high-density, multi-paragraph, professional-grade prompt.\n"
            "The final output must be 2x to 3x more detailed than a standard draft.\n\n"
            f"Original Task Keyword: {request.task}\n\n"
            f"EXPLICIT USER ANSWERS (USE EVERY ONE):\n{formatted_answers}\n"
            "--- END DIRECTIVE ---"
        )
        effective_task = directive
        
    # 2. Run Normal Generation
    result = orchestrator.run(user_task=effective_task, answers=request.answers)
    
    # 3. Standard Response Mapping
    api_trace = []
    raw_trace = getattr(result, "trace", [])
    for step in raw_trace:
        s_name = step.get("step_name") if isinstance(step, dict) else getattr(step, "step_name", "Unknown")
        s_out = step.get("output") if isinstance(step, dict) else getattr(step, "output", "")
        api_trace.append(PipelineStep(step_name=str(s_name), status="completed", output=str(s_out)))

    return AgentResponse(
        status="completed",
        final_prompt=getattr(result, "prompt", "No prompt returned"),
        execution_trace=api_trace,
        critique_score=getattr(result, "critique_score", None)
    )

