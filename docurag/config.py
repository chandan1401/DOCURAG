"""DocuRAG configuration settings module."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Attempt to load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _bool_from_env(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _int_from_env(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def _float_from_env(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val.strip())
    except ValueError:
        return default


@dataclass
class Settings:
    """Central configuration class for DocuRAG."""

    # LLM & Vision
    GEMINI_API_KEY: Optional[str] = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
    )
    GEMINI_MODEL: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    )

    # Storage Paths
    DOCURAG_STORAGE: str = field(
        default_factory=lambda: os.getenv("DOCURAG_STORAGE", "./storage")
    )

    # Embeddings & Reranking
    EMBED_MODEL: str = field(
        default_factory=lambda: os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    )
    RERANK_MODEL: str = field(
        default_factory=lambda: os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
    )

    # Ingestion & Vision Thresholds
    OCR_MIN_CHARS: int = field(
        default_factory=lambda: _int_from_env("OCR_MIN_CHARS", 50)
    )
    RENDER_DPI: int = field(
        default_factory=lambda: _int_from_env("RENDER_DPI", 200)
    )
    MIN_IMG_PX: int = field(
        default_factory=lambda: _int_from_env("MIN_IMG_PX", 150)
    )

    # Chunking Parameters
    CHUNK_TOKENS: int = field(
        default_factory=lambda: _int_from_env("CHUNK_TOKENS", 400)
    )
    CHUNK_OVERLAP: int = field(
        default_factory=lambda: _int_from_env("CHUNK_OVERLAP", 50)
    )

    # Retrieval & Ranking Parameters
    DENSE_K: int = field(
        default_factory=lambda: _int_from_env("DENSE_K", 30)
    )
    BM25_K: int = field(
        default_factory=lambda: _int_from_env("BM25_K", 30)
    )
    RERANK_IN: int = field(
        default_factory=lambda: _int_from_env("RERANK_IN", 20)
    )
    FINAL_K: int = field(
        default_factory=lambda: _int_from_env("FINAL_K", 6)
    )
    RRF_K: int = field(
        default_factory=lambda: _int_from_env("RRF_K", 60)
    )

    # Verification Thresholds
    SUPPORT_THRESHOLD: float = field(
        default_factory=lambda: _float_from_env("SUPPORT_THRESHOLD", 0.7)
    )

    # Runtime Modes & Agent Flags
    USE_AGENTS: bool = field(
        default_factory=lambda: _bool_from_env("USE_AGENTS", False)
    )
    MCP_MODE: str = field(
        default_factory=lambda: os.getenv("MCP_MODE", "direct").lower()
    )
    DOCURAG_FAKES: bool = field(
        default_factory=lambda: _bool_from_env("DOCURAG_FAKES", False)
    )
    NGROK_AUTHTOKEN: Optional[str] = field(
        default_factory=lambda: os.getenv("NGROK_AUTHTOKEN")
    )

    @property
    def storage_dir(self) -> Path:
        return Path(self.DOCURAG_STORAGE).resolve()

    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def pages_dir(self) -> Path:
        return self.storage_dir / "pages"

    @property
    def figs_dir(self) -> Path:
        return self.storage_dir / "figs"

    @property
    def index_dir(self) -> Path:
        return self.storage_dir / "index"

    @property
    def cache_dir(self) -> Path:
        return self.storage_dir / "cache"

    def ensure_dirs(self) -> None:
        """Create all storage directories if they do not exist."""
        for path in (
            self.storage_dir,
            self.uploads_dir,
            self.pages_dir,
            self.figs_dir,
            self.index_dir,
            self.cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


_global_settings: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """Retrieve the cached application Settings instance, optionally reloading."""
    global _global_settings
    if _global_settings is None or reload:
        _global_settings = Settings()
    return _global_settings
