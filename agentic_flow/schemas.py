from typing import Dict, List, Optional, Literal, Any
from pydantic import BaseModel

class PipelineStep(BaseModel):
    step_name: str
    status: str
    output: str

class AgentRequest(BaseModel):
    task: str
    model: Literal["gpt-4o", "gpt-4o-mini"] = "gpt-4o-mini"
    interactive_mode: bool = False
    answers: Optional[Dict[str, str]] = None

class AgentResponse(BaseModel):
    status: str
    prompt: Optional[str] = None
    final_prompt: Optional[str] = None
    questions: Optional[List[str]] = None
    execution_trace: Optional[List[PipelineStep]] = None
    critique_score: Optional[float] = None
