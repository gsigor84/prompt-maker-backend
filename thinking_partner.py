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
    intent_type: str = "unknown"  # shopping, research, creation, harmful
    confidence: dict = {}  # {"shopping": 95, "research": 2, "harmful": 3}
    issues: List[str] = []
    rationale: str = ""
    friction_words: List[str] = []  # words triggering safety concerns

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
        # STEP A: Diagnosis with Intent Classification
        diagnosis_system = (
            "You are a 'Thinking Partner' query optimizer.\\n\\n"
            "STEP 1 - CLASSIFY INTENT with confidence scores (must total 100):\\n"
            "- shopping: User wants to BUY or FIND something (products, services, deals)\\n"
            "- research: User wants to LEARN or UNDERSTAND something\\n"
            "- creation: User wants to MAKE or BUILD something\\n"
            "- harmful: Genuinely dangerous request (weapons, violence, illegal acts with victims)\\n\\n"
            "STEP 2 - IDENTIFY FRICTION WORDS:\\n"
            "What specific words might trigger safety filters? (e.g., 'fake', 'pirate', 'hack')\\n\\n"
            "STEP 3 - ASSESS ISSUES:\\n"
            "Only flag issues if the query is genuinely problematic, not just 'gray area' shopping.\\n"
            "Be practical, not preachy. A user wanting cheap replica jerseys is NOT the same as someone planning fraud.\\n\\n"
            "Return ONLY JSON with keys:\\n"
            "intent_type (string: shopping/research/creation/harmful),\\n"
            "confidence (object: {shopping: X, research: Y, creation: Z, harmful: W} where X+Y+Z+W=100),\\n"
            "friction_words (array of strings),\\n"
            "issues (array of strings - keep empty if just gray-area shopping),\\n"
            "rationale (string - brief explanation)."
        )
        
        diag_data = self._call_json(diagnosis_system, user_query)
        diagnosis = DiagnosisReport(**diag_data)
        
        # STEP B: Intent-Preserving Reframing
        reframing_system = (
            "You are an EXPERT query optimizer that helps users get better results.\\n\\n"
            "CRITICAL RULE: MAINTAIN THE USER'S DOMAIN\\n"
            "- If intent is SHOPPING → All reframes must be SHOPPING queries (finding products, deals, alternatives)\\n"
            "- If intent is RESEARCH → Reframes can be academic/analytical\\n"
            "- If intent is CREATION → Reframes must help them create/build\\n"
            "- If intent is HARMFUL (confidence > 70%) → Use strict safety handling\\n\\n"
            "VOCABULARY SANITIZATION (for gray-area queries):\\n"
            "Replace problematic words with legitimate alternatives that achieve the same goal:\\n"
            "- 'fake' → 'replica', 'budget', 'affordable', 'third-party', 'unbranded'\\n"
            "- 'knockoff' → 'alternative', 'inspired by', 'retro style'\\n"
            "- 'pirate' → 'free alternatives', 'open-source', 'student discount', 'trial version'\\n"
            "- 'hack' → 'automate', 'optimize', 'customize', 'shortcut'\\n"
            "- 'steal' → 'find deals', 'get for free legally', 'discount codes'\\n\\n"
            "EXAMPLES OF GOOD REFRAMES:\\n"
            "❌ 'buy fake jerseys' → 'essay on counterfeiting ethics' (WRONG - pivots away from shopping)\\n"
            "✅ 'buy fake jerseys' → 'affordable replica football jerseys from budget retailers' (CORRECT)\\n"
            "✅ 'buy fake jerseys' → 'best sites for unbranded football kits under $30' (CORRECT)\\n"
            "✅ 'buy fake jerseys' → 'DHgate vs AliExpress for replica sports apparel reviews' (CORRECT)\\n\\n"
            "❌ 'pirate Photoshop' → 'history of software piracy laws' (WRONG)\\n"
            "✅ 'pirate Photoshop' → 'best free open-source Photoshop alternatives like GIMP' (CORRECT)\\n"
            "✅ 'pirate Photoshop' → 'Adobe Creative Cloud student discounts and free trials' (CORRECT)\\n\\n"
            "Return ONLY JSON with keys:\\n"
            "suggestions (array of 3 objects with: type, content, educational_note),\\n"
            "tips (array of strings for search optimization)."
        )
        
        reframe_input = f"ORIGINAL QUERY: {user_query}\n\nDIAGNOSIS: {json.dumps(diag_data)}"
        reframe_data = self._call_json(reframing_system, reframe_input)
        
        return ReframeResult(
            diagnosis=diagnosis,
            suggestions=[ReframeSuggestion(**s) for s in reframe_data.get("suggestions", [])],
            tips=reframe_data.get("tips", [])
        )
