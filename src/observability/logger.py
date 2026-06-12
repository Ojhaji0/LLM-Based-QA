"""
Observability: Loguru logger + per-query stage tracer.
"""
import time
from loguru import logger

# Configure file rotation
logger.add(
    "logs/rag_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    format="{time} | {level} | {message}",
    level="INFO"
)


class QueryTracer:
    """Records timing for each pipeline stage and logs on completion."""

    def __init__(self, query: str):
        self.query = query
        self.timings: dict[str, float] = {}
        self._t0 = time.time()
        self._stage_start = self._t0

    def record(self, stage: str):
        now = time.time()
        self.timings[stage] = round((now - self._stage_start) * 1000, 1)
        self._stage_start = now

    def log_result(self, answer: str, n_chunks: int, citation_report: dict):
        total_ms = round((time.time() - self._t0) * 1000, 1)
        logger.info(
            f"QUERY | total={total_ms}ms | chunks={n_chunks} "
            f"| bad_citations={citation_report.get('hallucinated_citations', 0)} "
            f"| stages={self.timings} "
            f"| q={self.query[:80]!r} "
            f"| a={answer[:80]!r}"
        )
