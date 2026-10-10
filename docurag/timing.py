"""Execution timing utilities for stage latency measurements."""

import time
from contextlib import contextmanager
from typing import Dict, Generator, Optional

from docurag.schemas import Timing


class StageTimer:
    """Context manager and tracker for measuring per-stage latency in milliseconds."""

    def __init__(self):
        self._stages: Dict[str, int] = {}
        self._start_time: float = time.perf_counter()
        self._end_time: Optional[float] = None

    def __enter__(self) -> "StageTimer":
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._end_time = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        """Context manager measuring time elapsed in a named stage."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = int(round((time.perf_counter() - t0) * 1000))
            self._stages[name] = self._stages.get(name, 0) + elapsed_ms

    def ms(self, name: str) -> int:
        """Return the cumulative elapsed milliseconds for a named stage."""
        return self._stages.get(name, 0)

    def total_ms(self) -> int:
        """Return total elapsed milliseconds across all stages or overall lifecycle."""
        if self._end_time is not None:
            overall = int(round((self._end_time - self._start_time) * 1000))
            stage_sum = sum(self._stages.values())
            return max(overall, stage_sum)
        stage_sum = sum(self._stages.values())
        elapsed_overall = int(round((time.perf_counter() - self._start_time) * 1000))
        return max(stage_sum, elapsed_overall)

    def to_timing(self) -> Timing:
        """Export timing measurements to a standard Pydantic Timing model."""
        retrieval = self.ms("retrieval") or self.ms("retrieve")
        rerank = self.ms("rerank") or self.ms("reranking")
        generation = self.ms("generation") or self.ms("generate")
        verification = self.ms("verification") or self.ms("verify")
        ingest = self.ms("ingest") or self.ms("ingestion")
        total = self.total_ms()

        return Timing(
            retrieval_ms=retrieval,
            rerank_ms=rerank,
            generation_ms=generation,
            verification_ms=verification,
            total_ms=total,
            ingest_ms=ingest,
        )
