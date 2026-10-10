"""Grounded answer generation, citation attribution, and claim verification."""

from docurag.generate.answer import generate_answer, synthesize_comparison
from docurag.generate.verify import verify_answer
from docurag.generate.confidence import compute_confidence
from docurag.generate.prompts import (
    FIGURE_DESCRIPTION_PROMPT,
    GROUNDED_ANSWER_PROMPT,
    VERIFIER_PROMPT,
)

__all__ = [
    "generate_answer",
    "synthesize_comparison",
    "verify_answer",
    "compute_confidence",
    "FIGURE_DESCRIPTION_PROMPT",
    "GROUNDED_ANSWER_PROMPT",
    "VERIFIER_PROMPT",
]
