"""DocuRAG: Multimodal Retrieval-Augmented Generation Engine with Multi-Agent Orchestration."""

__version__ = "0.1.0"

from docurag.config import Settings, get_settings
from docurag.schemas import (
    AskResult,
    Chunk,
    ChunkHit,
    Confidence,
    ContentType,
    Evidence,
    FigureData,
    FigureDescription,
    Filters,
    GeneratedAnswer,
    PageData,
    RouteDecision,
    Source,
    SubQuestion,
    Timing,
    TraceStep,
    VerifierClaim,
    VerifierResult,
)
from docurag.pipeline import ask, ingest_file, ingest_directory

__all__ = [
    "Settings",
    "get_settings",
    "Chunk",
    "ChunkHit",
    "Evidence",
    "Source",
    "Confidence",
    "Timing",
    "TraceStep",
    "AskResult",
    "ContentType",
    "FigureData",
    "FigureDescription",
    "PageData",
    "RouteDecision",
    "SubQuestion",
    "GeneratedAnswer",
    "VerifierClaim",
    "VerifierResult",
    "Filters",
    "ask",
    "ingest_file",
    "ingest_directory",
]
