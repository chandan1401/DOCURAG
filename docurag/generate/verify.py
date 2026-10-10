"""Atomic claim verification and hallucination auditing."""

import os
import json
import logging
from typing import List, Union

from docurag.config import get_settings
from docurag.schemas import Evidence, GeneratedAnswer, VerifierResult
from docurag.generate.prompts import VERIFIER_PROMPT

logger = logging.getLogger(__name__)

def _is_fakes() -> bool:
    if os.getenv("DOCURAG_FAKES") in ("1", "true", "True"):
        return True
    try:
        return get_settings().DOCURAG_FAKES
    except Exception:
        return False

def _strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()

def verify_answer(
    answer: Union[GeneratedAnswer, str],
    evidence: List[Evidence],
) -> VerifierResult:
    """Decompose answer into atomic claims and verify each claim against cited evidence.

    Args:
        answer: GeneratedAnswer model or plain text answer.
        evidence: Evidence items cited in the answer.

    Returns:
        VerifierResult containing overall support_ratio and claim-by-claim breakdown.
    """
    if _is_fakes():
        return VerifierResult(
            support_ratio=1.0,
            claims=[
                {
                    "claim": "Answer is fully grounded in retrieved evidence.",
                    "supported": True,
                    "reason": "Direct textual match.",
                }
            ],
            supported=True
        )

    # Extract answer text
    answer_text = answer.answer if isinstance(answer, GeneratedAnswer) else str(answer)
    
    # Filter evidence to only those cited
    cited_eids = set()
    if isinstance(answer, GeneratedAnswer):
        cited_eids = set(answer.citations)
    else:
        # Simple heuristic for plain text
        import re
        cited_eids = set(re.findall(r"\[(E\d+)\]", answer_text))
    
    cited_evidence = [e for e in evidence if e.eid in cited_eids]
    
    # Build evidence blocks for verifier
    blocks = []
    for ev in cited_evidence:
        block = f"[{ev.eid}] {ev.chunk.source} p.{ev.chunk.page} ({ev.chunk.content_type}):\n{ev.chunk.content}"
        blocks.append(block)
    
    evidence_text = "\n\n".join(blocks)
    prompt = VERIFIER_PROMPT.format(answer=answer_text, cited_evidence_blocks=evidence_text)

    try:
        from google import genai
        settings = get_settings()
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not configured")
        
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=[prompt],
        )
        
        raw_text = getattr(response, "text", "") or ""
        json_str = _strip_json_fence(raw_text)
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Repair attempt
            start = json_str.find('{')
            end = json_str.rfind('}')
            if start != -1 and end != -1:
                try:
                    data = json.loads(json_str[start:end+1])
                except json.JSONDecodeError:
                    return VerifierResult(support_ratio=0.0, supported=False, claims=[])
            else:
                return VerifierResult(support_ratio=0.0, supported=False, claims=[])

        claims = data.get("claims", [])
        if not claims:
            return VerifierResult(support_ratio=0.0, supported=False, claims=[])

        supported_count = sum(1 for c in claims if c.get("supported") is True)
        support_ratio = supported_count / len(claims)
        
        return VerifierResult(
            claims=claims,
            support_ratio=support_ratio,
            supported=support_ratio > 0.5
        )

    except Exception as e:
        logger.error("Verification failed: %s", e)
        return VerifierResult(support_ratio=0.0, supported=False, claims=[])
