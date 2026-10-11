"""RAGAS automated evaluation harness runner.

Phase 1.2 — Evaluation Harness Foundation
Executes sequential evaluation with InMemoryRateLimiter (12 RPM) using Gemini 3.5 Flash-Lite judge.

Data flow (shared by Tier 1 baseline in Phase 3 and the Tier 2 ablations in Phase 14)::

    GoldenQA (dataset_builder) ──► pipeline(question, filters) ──► EvalSample
        ──► evaluate_samples() [RAGAS, 5 metrics, strictly sequential] ──► RagasReport

Any pipeline is plugged in as an async callable ``(question, filters) -> PipelineOutput``;
the harness itself never imports the pipelines, so it is usable from day one.
"""

from __future__ import annotations

import asyncio
import logging
import math
import sys
import types
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping, Sequence

from amorphnet.config import Settings, get_settings
from amorphnet.evaluation.dataset_builder import GoldenQA

logger = logging.getLogger(__name__)

JUDGE_MODEL = "gemini-3.5-flash-lite"
EMBEDDING_MODEL = "voyage-4-large"

# 12 RPM (one request / 5 s): safely below the 15 RPM Gemini free-tier ceiling.
JUDGE_REQUESTS_PER_SECOND = 0.2

# Canonical metric keys (architecture Section 15.2) and their hard targets.
METRIC_TARGETS: dict[str, float] = {
    "context_precision": 0.75,
    "context_recall": 0.80,
    "faithfulness": 0.90,
    "answer_relevance": 0.80,
    "answer_correctness": 0.75,
}

# RAGAS result-column name fragments -> canonical metric key.
_RAGAS_COLUMN_TO_CANONICAL: tuple[tuple[str, str], ...] = (
    ("context_precision", "context_precision"),
    ("context_recall", "context_recall"),
    ("faithfulness", "faithfulness"),
    ("answer_relevancy", "answer_relevance"),
    ("response_relevancy", "answer_relevance"),
    ("answer_correctness", "answer_correctness"),
)


# -----------------------------------------------------------------------------
# Data contracts
# -----------------------------------------------------------------------------
@dataclass
class PipelineOutput:
    """What an evaluated RAG pipeline must return for one question."""

    answer: str
    contexts: list[str]


# (question, filters) -> PipelineOutput. ``filters`` mirrors the /query API filter dict.
PipelineFn = Callable[[str, Mapping[str, Any]], Awaitable[PipelineOutput]]


@dataclass
class EvalSample:
    """One row of the RAGAS evaluation dataframe (question, contexts, answer, ground truth)."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    query_type: str = "unknown"


@dataclass
class RagasReport:
    """Aggregated + per-sample RAGAS scores for one evaluation run."""

    pipeline: str
    mode: str
    num_samples: int
    metrics: dict[str, float | None]
    by_query_type: dict[str, dict[str, float | None]]
    targets: dict[str, float]
    targets_met: dict[str, bool | None]
    per_sample: list[dict[str, Any]]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "mode": self.mode,
            "timestamp": self.timestamp,
            "num_samples": self.num_samples,
            "metrics": self.metrics,
            "targets": self.targets,
            "targets_met": self.targets_met,
            "by_query_type": self.by_query_type,
            "per_sample": self.per_sample,
        }


# -----------------------------------------------------------------------------
# RAGAS / LangChain compatibility
# -----------------------------------------------------------------------------
def ensure_ragas_compat() -> None:
    """Make ``import ragas`` work against ``langchain-community >= 0.4``.

    ragas 0.4.3 unconditionally imports ``ChatVertexAI`` from
    ``langchain_community.chat_models.vertexai``, a module that no longer ships in
    recent langchain-community releases. We never use Vertex AI (the judge is Gemini
    via ``langchain-google-genai``), so a placeholder module is registered only when the
    real one is missing.
    """
    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except ImportError:
        pass
    shim = types.ModuleType(name)

    class ChatVertexAI:  # noqa: D401 - placeholder, never instantiated
        """Placeholder for the removed langchain-community Vertex AI chat model."""

    shim.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
    sys.modules[name] = shim

    # ragas also does ``from langchain_community.llms import VertexAI``.
    try:
        import langchain_community.llms as llms_pkg

        if not hasattr(llms_pkg, "VertexAI"):
            try:
                llms_pkg.VertexAI = ChatVertexAI  # type: ignore[attr-defined]
            except Exception:  # lazy-loader modules may reject setattr
                pass
    except ImportError:
        pass


# -----------------------------------------------------------------------------
# Judge / embeddings / metrics construction
# -----------------------------------------------------------------------------
def build_run_config() -> Any:
    """Strictly sequential execution: no concurrency bursts against 15 RPM."""
    ensure_ragas_compat()
    from ragas.run_config import RunConfig

    return RunConfig(max_workers=1, timeout=60, max_retries=5, max_wait=60)


def build_judge_llm(settings: Settings | None = None) -> Any:
    """Gemini 3.5 Flash-Lite judge, rate-limited to 12 RPM, wrapped for RAGAS."""
    ensure_ragas_compat()
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas.llms import LangchainLLMWrapper

    cfg = settings or get_settings()
    if not cfg.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not configured; cannot build the RAGAS judge LLM.")

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=JUDGE_REQUESTS_PER_SECOND,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )
    chat = ChatGoogleGenerativeAI(
        model=JUDGE_MODEL,
        temperature=0.0,
        rate_limiter=rate_limiter,
        max_retries=5,
        google_api_key=cfg.GOOGLE_API_KEY,
    )
    return LangchainLLMWrapper(chat)


def build_judge_embeddings(settings: Settings | None = None) -> Any:
    """Voyage ``voyage-4-large`` embeddings wrapped for RAGAS."""
    ensure_ragas_compat()
    from langchain_voyageai import VoyageAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    cfg = settings or get_settings()
    if not cfg.VOYAGE_API_KEY:
        raise ValueError("VOYAGE_API_KEY is not configured; cannot build RAGAS embeddings.")
    return LangchainEmbeddingsWrapper(
        VoyageAIEmbeddings(model=EMBEDDING_MODEL, voyage_api_key=cfg.VOYAGE_API_KEY)
    )


def build_metrics() -> list[Any]:
    """Core RAGAS v0.4 metric instances (see architecture Section 15.3)."""
    ensure_ragas_compat()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import (
            AnswerCorrectness,
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )

    return [
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        Faithfulness(),
        ResponseRelevancy(),
        AnswerCorrectness(),
    ]


# -----------------------------------------------------------------------------
# Sample generation (pipeline -> EvalSample)
# -----------------------------------------------------------------------------
async def generate_samples(
    pipeline: PipelineFn,
    items: Sequence[GoldenQA],
    *,
    on_progress: Callable[[int, int, GoldenQA], None] | None = None,
) -> list[EvalSample]:
    """Run ``pipeline`` over golden items sequentially and collect RAGAS inputs.

    A pipeline failure on one question is recorded as an empty answer (scores 0 for
    that row) rather than aborting an expensive run.
    """
    samples: list[EvalSample] = []
    for idx, item in enumerate(items, start=1):
        try:
            out = await pipeline(item.question, item.filters)
            answer, contexts = out.answer, list(out.contexts)
        except Exception as err:  # noqa: BLE001
            logger.error("Pipeline failed on question %d (%s): %s", idx, item.query_type, err)
            answer, contexts = "", []
        samples.append(
            EvalSample(
                question=item.question,
                answer=answer,
                contexts=contexts,
                ground_truth=item.ground_truth,
                query_type=item.query_type,
            )
        )
        if on_progress:
            on_progress(idx, len(items), item)
    return samples


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def _mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return round(sum(clean) / len(clean), 4) if clean else None


def _canonicalise(raw_scores: Mapping[str, Any]) -> dict[str, float | None]:
    row: dict[str, float | None] = {}
    for column, value in raw_scores.items():
        for fragment, canonical in _RAGAS_COLUMN_TO_CANONICAL:
            if fragment in column:
                try:
                    num = float(value)
                except (TypeError, ValueError):
                    num = float("nan")
                row[canonical] = None if math.isnan(num) else num
                break
    return row


def build_report(
    samples: Sequence[EvalSample],
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    pipeline: str,
    mode: str,
) -> RagasReport:
    """Aggregate per-sample RAGAS rows into overall / per-query-type means and target checks."""
    if len(samples) != len(raw_rows):
        raise ValueError(f"RAGAS returned {len(raw_rows)} rows for {len(samples)} samples.")

    rows = [_canonicalise(r) for r in raw_rows]
    per_sample = [
        {"question": s.question, "query_type": s.query_type, "scores": row}
        for s, row in zip(samples, rows)
    ]

    metrics = {m: _mean([r.get(m) for r in rows]) for m in METRIC_TARGETS}  # type: ignore[misc]

    grouped: dict[str, list[dict[str, float | None]]] = defaultdict(list)
    for s, row in zip(samples, rows):
        grouped[s.query_type].append(row)
    by_type = {
        qt: {m: _mean([r.get(m) for r in rs]) for m in METRIC_TARGETS}  # type: ignore[misc]
        for qt, rs in grouped.items()
    }

    targets_met = {
        m: (None if metrics[m] is None else metrics[m] >= target)  # type: ignore[operator]
        for m, target in METRIC_TARGETS.items()
    }
    return RagasReport(
        pipeline=pipeline,
        mode=mode,
        num_samples=len(samples),
        metrics=metrics,
        by_query_type=by_type,
        targets=dict(METRIC_TARGETS),
        targets_met=targets_met,
        per_sample=per_sample,
    )


def evaluate_samples(
    samples: Sequence[EvalSample],
    *,
    pipeline: str = "unknown",
    mode: str = "smoke",
    settings: Settings | None = None,
    show_progress: bool = True,
) -> RagasReport:
    """Score ``samples`` with the five RAGAS metrics (synchronous, strictly sequential)."""
    if not samples:
        raise ValueError("No samples to evaluate.")
    ensure_ragas_compat()
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=s.question,
                retrieved_contexts=s.contexts,
                response=s.answer,
                reference=s.ground_truth,
            )
            for s in samples
        ]
    )
    result = evaluate(
        dataset=dataset,
        metrics=build_metrics(),
        llm=build_judge_llm(settings),
        embeddings=build_judge_embeddings(settings),
        run_config=build_run_config(),
        show_progress=show_progress,
    )
    return build_report(samples, result.scores, pipeline=pipeline, mode=mode)


async def aevaluate_samples(samples: Sequence[EvalSample], **kwargs: Any) -> RagasReport:
    """Async wrapper (used by the ``POST /evaluate`` route); runs RAGAS in a worker thread."""
    return await asyncio.to_thread(evaluate_samples, samples, **kwargs)
