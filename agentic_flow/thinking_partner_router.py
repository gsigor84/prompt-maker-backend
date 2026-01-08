from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
import os
from datetime import datetime, timezone
from thinking_partner import ThinkingPartnerService, ReframeResult

router = APIRouter(prefix="/api/v2/thinking-partner", tags=["Thinking Partner"])

class AnalysisRequest(BaseModel):
    query: str

class FeedbackRequest(BaseModel):
    query: str
    reframe_helpful: bool
    actual_intent: str  # shopping, research, creation, other
    user_correction: Optional[str] = None

@router.post("/analyze", response_model=ReframeResult)
async def analyze_query(request: AnalysisRequest):
    """
    Diagnoses and reframes a user query with intent-aware sanitization.
    """
    try:
        service = ThinkingPartnerService()
        result = service.analyze(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Records user feedback to improve reframing quality over time.
    """
    feedback_path = "data/feedback.jsonl"
    os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
    
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "query": request.query,
        "reframe_helpful": request.reframe_helpful,
        "actual_intent": request.actual_intent,
        "user_correction": request.user_correction
    }
    
    with open(feedback_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    return {"status": "recorded", "message": "Thank you for your feedback!"}

