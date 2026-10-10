"""Confidence scoring heuristic combining relevance, support, and citation coverage."""

from typing import List, Union

from docurag.schemas import Confidence, Evidence, GeneratedAnswer

def compute_confidence(
    evidence: List[Evidence],
    support_ratio: float,
    answer: Union[GeneratedAnswer, str],
) -> Confidence:
    """Compute holistic confidence score and classification label.

    Formula:
        score = 0.4 * top_norm_score + 0.4 * support_ratio + 0.2 * coverage
        label = 'High' (>= 0.8), 'Medium' (>= 0.5), else 'Low'.
        Capped at 0.2 with label 'Low' if unanswerable.
    """
    answer_text = answer.answer if isinstance(answer, GeneratedAnswer) else str(answer)
    is_answerable = answer.answerable if isinstance(answer, GeneratedAnswer) else True

    if not is_answerable or not evidence:
        return Confidence(
            score=0.2,
            label="Low",
            support_ratio=support_ratio,
            citation_coverage=0.0,
            reason="Insufficient evidence to provide grounded answer.",
        )

    # Use norm_score as requested
    top_score = max((e.norm_score for e in evidence), default=0.5)
    
    # Simple coverage: check if any citation [E#] exists in text
    # Better coverage would be (count of unique citations used / total evidence provided)
    import re
    citations_found = set(re.findall(r"\[(E\d+)\]", answer_text))
    evidence_ids = {e.eid for e in evidence}
    
    # Coverage = proportion of evidence used
    coverage = len(citations_found.intersection(evidence_ids)) / len(evidence_ids) if evidence_ids else 0.0

    score = 0.4 * top_score + 0.4 * support_ratio + 0.2 * coverage
    score = round(min(1.0, max(0.0, score)), 2)

    if score >= 0.8:
        label = "High"
    elif score >= 0.5:
        label = "Medium"
    else:
        label = "Low"

    return Confidence(
        score=score,
        label=label,
        support_ratio=support_ratio,
        citation_coverage=coverage,
        reason=f"Score based on top relevance {top_score:.2f}, support {support_ratio:.2f}, and coverage {coverage:.2f}",
    )
