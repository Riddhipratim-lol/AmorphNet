"""Unit tests for Phase 1: golden dataset utilities and RAGAS report aggregation."""

from __future__ import annotations

import json

import pytest

from amorphnet.evaluation.dataset_builder import (
    SMOKE_QUOTAS,
    SMOKE_SIZE,
    distribution,
    get_eval_subset,
    load_golden_dataset,
    select_smoke_subset,
    validate_dataset,
)
from amorphnet.evaluation.ragas_runner import (
    METRIC_TARGETS,
    EvalSample,
    PipelineOutput,
    build_report,
    generate_samples,
)


def test_golden_dataset_valid():
    items = load_golden_dataset()
    assert len(items) == 50
    assert validate_dataset(items) == []
    for item in items:
        assert set(item.filters) == {"company_name", "fiscal_year", "filing_type"}


def test_smoke_subset_is_stratified_and_deterministic():
    items = load_golden_dataset()
    subset = select_smoke_subset(items)
    assert len(subset) == SMOKE_SIZE
    assert distribution(subset) == SMOKE_QUOTAS
    assert [s.question for s in subset] == [s.question for s in select_smoke_subset(items)]
    assert len(get_eval_subset("full")) == 50
    assert len(get_eval_subset("smoke")) == SMOKE_SIZE


def test_validate_dataset_flags_problems():
    items = load_golden_dataset()
    assert validate_dataset(items[:-1])  # wrong size
    assert validate_dataset(items + [items[0]], expected_size=51)  # duplicate question


def test_build_report_aggregation():
    samples = [
        EvalSample("q1", "a", ["c"], "gt", "factual_numeric"),
        EvalSample("q2", "a", ["c"], "gt", "semantic_risk"),
    ]
    rows = [
        {
            "llm_context_precision_with_reference": 1.0,
            "context_recall": 1.0,
            "faithfulness": 1.0,
            "answer_relevancy": 0.9,
            "answer_correctness": 0.8,
        },
        {
            "llm_context_precision_with_reference": 0.5,
            "context_recall": float("nan"),
            "faithfulness": 0.8,
            "answer_relevancy": 0.7,
            "answer_correctness": 0.4,
        },
    ]
    report = build_report(samples, rows, pipeline="baseline", mode="smoke")
    assert report.metrics["context_precision"] == 0.75
    assert report.metrics["context_recall"] == 1.0  # NaN ignored
    assert report.metrics["faithfulness"] == 0.9
    assert report.targets_met["faithfulness"] is True
    assert report.targets_met["answer_correctness"] is False
    assert set(report.by_query_type) == {"factual_numeric", "semantic_risk"}
    assert set(report.metrics) == set(METRIC_TARGETS)
    json.dumps(report.to_dict())  # serialisable

    with pytest.raises(ValueError):
        build_report(samples, rows[:1], pipeline="x", mode="smoke")


@pytest.mark.asyncio
async def test_generate_samples_isolates_pipeline_failures():
    items = get_eval_subset("smoke")
    calls: list[dict] = []

    async def pipeline(question, filters):
        calls.append(dict(filters))
        if len(calls) == 2:
            raise RuntimeError("boom")
        return PipelineOutput(answer="ok", contexts=["ctx"])

    samples = await generate_samples(pipeline, items)
    assert len(samples) == len(items)
    assert samples[1].answer == "" and samples[1].contexts == []
    assert samples[0].answer == "ok"
    assert calls[0]["fiscal_year"] == 2025
