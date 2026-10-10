"""Grounded answer generation and comparison synthesis."""

import os
import re
import json
import time
import logging
from typing import Any, Dict, List, Optional, Union

from docurag.config import get_settings
from docurag.schemas import Evidence, GeneratedAnswer
from docurag.generate.prompts import GROUNDED_ANSWER_PROMPT

logger = logging.getLogger(__name__)

UNAVAILABLE_MSG = (
    "The answer service is temporarily unavailable. Please try again in a moment."
)


def _is_fakes() -> bool:
    if os.getenv("DOCURAG_FAKES") in ("1", "true", "True"):
        return True
    try:
        return get_settings().DOCURAG_FAKES
    except Exception:
        return False


def _strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    # Prefer an explicit ```json fenced block
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Otherwise take the outermost {...} object
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


def _generate_with_gemini(prompt: str, settings) -> Optional[str]:
    """Try Gemini up to 3 times. Returns text or None."""
    gemini_key = getattr(settings, "GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        logger.warning("GEMINI_API_KEY is not set; skipping Gemini.")
        return None

    from google import genai

    client = genai.Client(api_key=gemini_key)
    model_name = getattr(settings, "GEMINI_MODEL", "gemini-3.5-flash") or "gemini-3.5-flash"

    for attempt in range(3):
        try:
            logger.info("Generating answer using Gemini model: %s (attempt %d)", model_name, attempt + 1)
            response = client.models.generate_content(model=model_name, contents=prompt)
            if response and response.text:
                return response.text
            logger.warning("Gemini attempt %d returned an empty response.", attempt + 1)
        except Exception as err:
            logger.warning("Gemini attempt %d failed: %s", attempt + 1, err)
        if attempt < 2:
            time.sleep(2 ** attempt)  # 1s, then 2s
    return None


def _generate_with_nvidia(prompt: str, settings) -> Optional[str]:
    """Fallback provider. Only used when NVIDIA_API_KEY is configured."""
    nvidia_key = getattr(settings, "NVIDIA_API_KEY", None) or os.getenv("NVIDIA_API_KEY")
    if not nvidia_key:
        return None

    import requests

    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {nvidia_key}",
        "Accept": "application/json",
    }
    payload = {
        "model": os.getenv("NVIDIA_MODEL", "google/diffusiongemma-26b-a4b-it"),
        "max_tokens": 1024,
        "stream": False,
        "temperature": 0.1,
        "top_p": 1,
        "messages": [{"role": "user", "content": prompt}],
    }

    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = requests.post(invoke_url, headers=headers, json=payload, timeout=(5, 25))
            if response.status_code in (429, 503) and attempt < max_retries - 1:
                time.sleep(2 * (2 ** attempt))
                continue
            response.raise_for_status()
            data = response.json()
            if data.get("choices"):
                return data["choices"][0]["message"]["content"]
            return None
        except Exception as err:
            logger.warning("NVIDIA attempt %d failed: %s", attempt + 1, err)
            if attempt < max_retries - 1:
                time.sleep(2)
    return None


def generate_answer(
    question: str,
    evidence: List[Evidence],
    instructions: Optional[str] = None,
) -> GeneratedAnswer:
    """Generate grounded answer strictly constrained to supplied evidence with citations.

    Args:
        question: User query.
        evidence: Retrieved and verified Evidence list.
        instructions: Optional specialist prompt instructions.

    Returns:
        GeneratedAnswer instance with answer text, citations, and answerable flag.
    """
    if _is_fakes() or not evidence:
        if not evidence:
            return GeneratedAnswer(
                answer="The available documents do not contain sufficient evidence to answer this question.",
                citations=[],
                answerable=False,
            )
        cited_eids = [e.eid for e in evidence[:2]]
        citations_str = "".join(f"[{eid}]" for eid in cited_eids)
        return GeneratedAnswer(
            answer=f"DocuRAG provides accurate grounded responses {citations_str}.",
            citations=cited_eids,
            answerable=True,
        )

    try:
        settings = get_settings()

        # Build evidence blocks: [E1] {source} p.{page} ({content_type}):\n{content}
        blocks = []
        for ev in evidence:
            blocks.append(
                f"[{ev.eid}] {ev.chunk.source} p.{ev.chunk.page} ({ev.chunk.content_type}):\n{ev.chunk.content}"
            )

        evidence_text = "\n\n".join(blocks)
        prompt = GROUNDED_ANSWER_PROMPT.format(question=question, evidence_blocks=evidence_text)
        if instructions:
            prompt = f"{instructions}\n\n{prompt}"

        # 1. Gemini (with retries), 2. NVIDIA fallback (only if key configured)
        raw_text = _generate_with_gemini(prompt, settings)
        if not raw_text:
            raw_text = _generate_with_nvidia(prompt, settings)

        if not raw_text:
            return GeneratedAnswer(
                answer=UNAVAILABLE_MSG,
                citations=[],
                answerable=False,
            )

        json_str = _strip_json_fence(raw_text)
        data = None

        # Robust JSON parsing
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            start = json_str.find("{")
            end = json_str.rfind("}")
            if start != -1 and end != -1:
                try:
                    data = json.loads(json_str[start:end + 1])
                except json.JSONDecodeError:
                    pass

        if not isinstance(data, dict):
            # Model returned markdown or free-form text instead of strict JSON
            valid_eids = [e.eid for e in evidence if f"[{e.eid}]" in raw_text]
            return GeneratedAnswer(
                answer=raw_text.strip(),
                citations=valid_eids,
                answerable=True,
            )

        # Validate and drop unknown citation IDs
        valid_eids = {e.eid for e in evidence}
        citations = [cid for cid in data.get("citations", []) if cid in valid_eids]
        final_answer = data.get("answer", "")
        if not final_answer:
            final_answer = "The documents provided do not contain a clear answer to the question."

        return GeneratedAnswer(
            answer=final_answer,
            citations=citations,
            answerable=data.get("answerable", True),
        )

    except Exception as e:
        logger.exception("Answer generation failed: %s", e)
        return GeneratedAnswer(
            answer=UNAVAILABLE_MSG,
            citations=[],
            answerable=False,
        )


def synthesize_comparison(
    question: str,
    sub_answers: Union[List[Dict[str, Any]], str],
) -> GeneratedAnswer:
    """Synthesize sub-answers into unified multi-document comparison answer."""
    # For M5, we keep this as a stub or implement a similar loop.
    # The requirement focuses on generate_answer.
    return GeneratedAnswer(
        answer=f"Comparison summary for '{question}' [E1][E2].",
        citations=["E1", "E2"],
        answerable=True,
    )