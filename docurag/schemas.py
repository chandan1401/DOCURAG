"""Data schemas and contracts for DocuRAG."""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator

ContentType = Literal["text", "table", "image", "ocr"]
ConfidenceLabel = Literal["High", "Medium", "Low"]
RouteIntent = Literal[
    "factual",
    "table_lookup",
    "chart_numeric",
    "multi_doc_compare",
    "summarize_doc",
    "out_of_scope",
]
QueryType = Literal["general", "table", "chart"]
FigureType = Literal[
    "bar_chart",
    "line_chart",
    "pie_chart",
    "table",
    "diagram",
    "photo",
    "scanned_text",
    "other",
]


class Chunk(BaseModel):
    """Atomic content unit extracted from an ingested document."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    chunk_id: str
    document_id: str = ""
    source: str = ""
    page: int = 1
    content_type: ContentType = "text"
    section: Optional[str] = None
    content: str = ""
    image_path: Optional[str] = None
    page_image_path: Optional[str] = None
    ocr_used: bool = False
    token_count: int = 0
    bbox: Optional[List[float]] = None

    @model_validator(mode="before")
    @classmethod
    def _remap_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "doc_id" in data and "document_id" not in data:
                data["document_id"] = data.get("doc_id", "")
            if "doc_name" in data and "source" not in data:
                data["source"] = data.get("doc_name", "")
            if "text" in data and "content" not in data:
                data["content"] = data.get("text", "")
            if "content_type" not in data:
                if data.get("is_table"):
                    data["content_type"] = "table"
                elif data.get("is_figure"):
                    data["content_type"] = "image"
            elif data.get("is_table") and data["content_type"] == "text":
                data["content_type"] = "table"
            elif data.get("is_figure") and data["content_type"] == "text":
                data["content_type"] = "image"
        return data

    @property
    def doc_id(self) -> str:
        return self.document_id

    @property
    def doc_name(self) -> str:
        return self.source

    @property
    def text(self) -> str:
        return self.content

    @property
    def is_table(self) -> bool:
        return self.content_type == "table"

    @property
    def is_figure(self) -> bool:
        return self.content_type == "image"


class Evidence(BaseModel):
    """Retrieved and reranked chunk associated with a specific query."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    eid: str = "E1"
    chunk: Chunk
    score: float = 0.0
    norm_score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float = 0.0
    combined_score: float = 0.0


class Source(BaseModel):
    """Attributed source citation rendered in the user interface."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str = "1"
    document: str = ""
    page: int = 1
    type: ContentType = "text"
    snippet: str = ""
    thumbnail_url: Optional[str] = None
    score: float = 0.0
    confidence: float = 0.0
    bbox: Optional[List[float]] = None

    @model_validator(mode="before")
    @classmethod
    def _remap_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "citation_id" in data and "id" not in data:
                data["id"] = str(data.get("citation_id"))
            if "doc_name" in data and "document" not in data:
                data["document"] = data.get("doc_name", "")
            if "quote" in data and "snippet" not in data:
                data["snippet"] = data.get("quote", "")
        return data

    @property
    def citation_id(self) -> int:
        try:
            return int(self.id)
        except ValueError:
            return 1

    @property
    def doc_name(self) -> str:
        return self.document

    @property
    def quote(self) -> str:
        return self.snippet


class Confidence(BaseModel):
    """Groundedness and confidence assessment for generated answer."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    score: float
    label: ConfidenceLabel = "High"
    support_ratio: float = 1.0
    citation_coverage: float = 1.0
    reason: str = ""

    @model_validator(mode="before")
    @classmethod
    def _infer_label(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "score" in data and "label" not in data:
                try:
                    s = float(data["score"])
                    if s >= 0.75:
                        data["label"] = "High"
                    elif s >= 0.4:
                        data["label"] = "Medium"
                    else:
                        data["label"] = "Low"
                except (ValueError, TypeError):
                    pass
        return data


class Timing(BaseModel):
    """Per-stage execution latency in milliseconds."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    retrieval_ms: int = 0
    rerank_ms: int = 0
    generation_ms: int = 0
    verification_ms: int = 0
    total_ms: int = 0
    ingest_ms: int = 0

    @model_validator(mode="before")
    @classmethod
    def _remap_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "retrieve_ms" in data and "retrieval_ms" not in data:
                data["retrieval_ms"] = data.get("retrieve_ms", 0)
            if "generate_ms" in data and "generation_ms" not in data:
                data["generation_ms"] = data.get("generate_ms", 0)
            if "verify_ms" in data and "verification_ms" not in data:
                data["verification_ms"] = data.get("verify_ms", 0)
        return data

    @property
    def retrieve_ms(self) -> int:
        return self.retrieval_ms

    @property
    def generate_ms(self) -> int:
        return self.generation_ms

    @property
    def verify_ms(self) -> int:
        return self.verification_ms


class TraceStep(BaseModel):
    """Step in multi-agent execution pipeline."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    agent: str
    action: str
    tool: Optional[str] = None
    detail: str = ""
    ms: int = 0


class AskResult(BaseModel):
    """Complete query response contract returned by the RAG pipeline."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    question: str = ""
    answer: str = ""
    answerable: bool = True
    sources: List[Source] = Field(default_factory=list)
    confidence: Confidence = Field(default_factory=lambda: Confidence(score=1.0, label="High"))
    evidence: List[Evidence] = Field(default_factory=list)
    timing: Timing = Field(default_factory=Timing)
    verified: bool = True
    support_ratio: float = 1.0
    agent_trace: List[TraceStep] = Field(default_factory=list)


# --- Ingestion, Vision & Agent Schemas ---

class FigureData(BaseModel):
    """Figure bounding box and saved file location extracted during ingestion."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    figure_id: str = ""
    image_path: str = ""
    page: int = 1
    bbox: Optional[List[float]] = None
    caption: Optional[str] = None
    description: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "path" in data and "image_path" not in data:
                data["image_path"] = data["path"]
            elif "image_path" in data and "path" not in data:
                data["path"] = data["image_path"]
        return data

    @property
    def path(self) -> str:
        return self.image_path


class FigureDescription(BaseModel):
    """Structured Gemini Vision indexing metadata for document figures."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    type: str = "other"
    title: str = ""
    caption: str = ""
    key_values: List[Dict[str, Any]] = Field(default_factory=list)
    relationships: str = ""
    visible_text: str = ""

    @model_validator(mode="before")
    @classmethod
    def _sanitize_key_values(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            key_values = data.get("key_values")
            if isinstance(key_values, list):
                data["key_values"] = [kv for kv in key_values if isinstance(kv, dict)]
            elif key_values is None:
                data["key_values"] = []
        return data

    def to_flattened_text(self) -> str:
        """Flatten into dense searchable text representation."""
        kv_pairs = "; ".join(
            f"{kv.get('label', '')}={kv.get('value', '')} {kv.get('unit', '')}".strip()
            for kv in self.key_values
            if isinstance(kv, dict) and kv.get("label")
        )
        kv_part = f" Values: {kv_pairs}." if kv_pairs else ""
        rel_part = f" Relationships: {self.relationships}." if self.relationships else ""
        text_part = f" Text: {self.visible_text}." if self.visible_text else ""
        caption_part = f" {self.caption}." if self.caption else ""
        return f"[{self.type}] {self.title}.{caption_part}{kv_part}{rel_part}{text_part}".strip()


class PageData(BaseModel):
    """Raw extracted page information before semantic chunking."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    page: int = 1
    text: str = ""
    ocr_used: bool = False
    tables_md: List[str] = Field(default_factory=list)
    figures: List[FigureData] = Field(default_factory=list)
    page_image_path: str = ""

    @model_validator(mode="before")
    @classmethod
    def _normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "page_number" in data and "page" not in data:
                data["page"] = data["page_number"]
            if "content" in data and "text" not in data:
                data["text"] = data["content"]
        return data


class RouteDecision(BaseModel):
    """Router Agent classification decision."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    intent: str = "factual"
    query_type: str = "general"
    document_hints: List[str] = Field(default_factory=list)
    needs_calculation: bool = False

    @model_validator(mode="before")
    @classmethod
    def _normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            valid_intents = {
                "factual",
                "table_lookup",
                "chart_numeric",
                "multi_doc_compare",
                "summarize_doc",
                "out_of_scope",
            }
            if "intent" in data and data["intent"] not in valid_intents:
                data["intent"] = "factual"

            valid_modalities = {"general", "table", "chart"}
            if "query_type" in data and data["query_type"] not in valid_modalities:
                data["query_type"] = "general"

            if isinstance(data.get("document_hints"), str):
                data["document_hints"] = [data["document_hints"]]
        return data


class SubQuestion(BaseModel):
    """Planner Agent decomposed sub-question."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    sub_question: str = ""
    document_names: List[str] = Field(default_factory=list)
    document_ids: Optional[List[str]] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "question" in data and "sub_question" not in data:
                data["sub_question"] = data["question"]
            if "document_ids" in data and not data.get("document_names"):
                ids = data["document_ids"]
                if isinstance(ids, list):
                    data["document_names"] = ids
            elif "document_names" in data and data.get("document_ids") is None:
                names = data["document_names"]
                if isinstance(names, list):
                    data["document_ids"] = names
        return data


class GeneratedAnswer(BaseModel):
    """Structured response from LLM answer generation."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    answer: str = ""
    citations: List[str] = Field(default_factory=list)
    answerable: bool = True


class VerifierClaim(BaseModel):
    """Atomic claim extracted and verified by Verifier Agent."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    claim: str = ""
    supported: bool = True
    citation: Optional[str] = None
    reason: str = ""


class VerifierResult(BaseModel):
    """Result of claim verification against cited evidence."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    claims: List[Dict[str, Any]] = Field(default_factory=list)
    support_ratio: float = 1.0
    supported: bool = True


class Filters(BaseModel):
    """Metadata filter configuration for retrieval."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    document_ids: Optional[List[str]] = None
    content_types: Optional[List[str]] = None


class ChunkHit(BaseModel):
    """Search hit dictionary shape used in FastMCP tool servers."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    chunk_id: str = ""
    document: str = ""
    page: int = 1
    content_type: str = "text"
    snippet: str = ""
    score: float = 0.0


__all__ = [
    "ContentType",
    "ConfidenceLabel",
    "RouteIntent",
    "QueryType",
    "FigureType",
    "Chunk",
    "Evidence",
    "Source",
    "Confidence",
    "Timing",
    "TraceStep",
    "AskResult",
    "FigureData",
    "FigureDescription",
    "PageData",
    "RouteDecision",
    "SubQuestion",
    "GeneratedAnswer",
    "VerifierClaim",
    "VerifierResult",
    "Filters",
    "ChunkHit",
]
