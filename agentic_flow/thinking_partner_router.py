from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from thinking_partner import ThinkingPartnerService, ReframeResult

router = APIRouter(prefix="/api/v2/thinking-partner", tags=["Thinking Partner"])

class AnalysisRequest(BaseModel):
    query: str

@router.post("/analyze", response_model=ReframeResult)
async def analyze_query(request: AnalysisRequest):
    """
    Diagnoses and reframes a user query into academic or sociological contexts.
    """
    try:
        service = ThinkingPartnerService()
        result = service.analyze(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
