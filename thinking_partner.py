from __future__ import annotations
import os
import json
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger(__name__)

# -------------------------
# Models
# -------------------------

class DiagnosisReport(BaseModel):
    issues: List[str]
    rationale: str

class ReframeSuggestion(BaseModel):
    type: str # e.g., "Academic", "Perspective Shift", "Learning"
    content: str # The new reframed prompt
    educational_note: str # Why this reframe is better

class ReframeResult(BaseModel):
    diagnosis: DiagnosisReport
    suggestions: List[ReframeSuggestion]
    tips: List[str]

# -------------------------
# Thinking Partner Service
# -------------------------

class ThinkingPartnerService:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            logger.warning("⚠️ OPENAI_API_KEY missing for Thinking Partner. Logic will fail.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
        
        self.model = model

    def _call_json(self, system_prompt: str, user_input: str) -> dict:
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Thinking Partner LLM Call failed: {e}")
            raise

    def analyze(self, user_query: str) -> ReframeResult:
        """
        Full 2-step pipeline: Diagnosis -> Reframing.
        """
        # STEP A: Diagnosis
        diagnosis_system = (
            "You are a 'Thinking Partner' diagnostic agent.\n"
            "Analyze the user's input for vague intent, loaded language, bias, or potential ethical/policy violations.\n"
            "Explain CLEARLY why these are issues for an AI or for critical thinking.\n"
            "CRITICAL: Be objective and analytical. Avoid being preachy. Focus on the structural reasons "
            "(e.g., 'facilitating crime' vs 'studying crime') rather than just moralizing.\n"
            "Return ONLY JSON with keys: issues (array of strings), rationale (string)."
        )
        
        diag_data = self._call_json(diagnosis_system, user_query)
        diagnosis = DiagnosisReport(**diag_data)
        
        # STEP B: Reframing
        reframing_system = (
            "You are an EXPERT critical thinking orchestrator.\n"
            "Based on the diagnosis provided, provide 3 distinct reframes of the original query.\n\n"
            "CRITICAL GOAL: TOPIC PRESERVATION.\n"
            "Do NOT switch the topic to a safe/legal alternative. Instead, reframe it into an ACADEMIC or SOCIOLOGICAL study of the ORIGINAL TOPIC.\n\n"
            "CRITICAL: PRIORITIZE RECENCY. Focus the reframes on the MOST RECENT documented trends, latest available seizure data (e.g., last 12-24 months), and contemporary geopolitical context. Avoid defaulting to 'historical' analysis.\n\n"
            "Available Frames:\n"
            "1. Academic/Neutral Frame: Focus on the phenomenon as a CURRENT study.\n"
            "2. Perspective Shift: Look at it from the lens of a researcher, journalist, or fictional world-builder.\n"
            "3. Learning/Prevention Frame: Turns 'how to do' into 'how it works structurally' or 'how to detect'.\n\n"
            "Return ONLY JSON with keys:\n"
            "suggestions (array of objects with: type, content, educational_note),\n"
            "tips (array of strings for general improvement)."
        )
        
        reframe_input = f"ORIGINAL QUERY: {user_query}\n\nDIAGNOSIS: {json.dumps(diag_data)}"
        reframe_data = self._call_json(reframing_system, reframe_input)
        
        return ReframeResult(
            diagnosis=diagnosis,
            suggestions=[ReframeSuggestion(**s) for s in reframe_data.get("suggestions", [])],
            tips=reframe_data.get("tips", [])
        )
