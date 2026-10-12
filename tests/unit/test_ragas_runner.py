"""Phase 1.2 — RAGAS harness construction (judge, metrics, run-config) and report maths."""

from __future__ import annotations

import math

import pytest

from amorphnet.config import Settings
from amorphnet.evaluation import ragas_runner as rr


def test_constants():
    assert rr.JUDGE_MODEL == "gemini-3.5-flash-lite"
    assert rr.EMBEDDING_MODEL == "voyage-4-large"
    assert rr.JUDGE_REQUESTS_PER_SECOND == pytest.approx(0.2)  # 12 RPM < 15 RPM ceiling
    assert rr.METRIC_TARGETS == {
        "context_precision": 0.75, "context_recall": 0.80, "faithfulness": 0.90,
        "answer_relevance": 0.80, "answer_correctness": 0.75,
    }


def test_run_config_is_strictly_sequential_with_retries():
    cfg = rr.build_run_config()
    assert (cfg.max_workers, cfg.timeout, cfg.max_retries, cfg.max_wait) == (1, 60, 5, 60)


def test_build_metrics_returns_the_five_core_metrics():
    names = {type(m).__name__ for m in rr.build_metrics()}
    assert names == {"LLMContextPrecisionWithReference", "LLMContextRecall", "Faithfulness",
                     "ResponseRelevancy", "AnswerCorrectness"}


def test_judge_requires_credentials(clean_env):
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        rr.build_judge_llm(Settings(_env_file=None))
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        rr.build_judge_embeddings(Settings(_env_file=None))


def test_judge_llm_is_rate_limited_gemini_flash_lite(clean_env):
    llm = rr.build_judge_llm(Settings(_env_file=None, GOOGLE_API_KEY="AIza-test"))
    chat = llm.langchain_llm
    assert chat.model.endswith("gemini-3.5-flash-lite")
    assert chat.temperature == 0.0
    assert chat.rate_limiter is not None
    assert chat.rate_limiter.requests_per_second == pytest.approx(0.2)
    assert chat.max_retries == 5


def test_ensure_ragas_compat_is_idempotent():
    rr.ensure_ragas_compat()
    rr.ensure_ragas_compat()
    import ragas  # noqa: F401


@pytest.mark.parametrize("raw,expected", [
    ({"llm_context_precision_with_reference": 0.5}, {"context_precision": 0.5}),
    ({"context_recall": "0.7"}, {"context_recall": 0.7}),
    ({"answer_relevancy": 0.4}, {"answer_relevance": 0.4}),
    ({"response_relevancy": 0.3}, {"answer_relevance": 0.3}),
    ({"faithfulness": float("nan")}, {"faithfulness": None}),
    ({"faithfulness": "oops"}, {"faithfulness": None}),
    ({"user_input": "ignored text"}, {}),
])
def test_canonicalise_maps_ragas_columns(raw, expected):
    assert rr._canonicalise(raw) == expected


def test_mean_ignores_nan_none_and_rounds():
    assert rr._mean([1.0, float("nan"), None, 0.5]) == 0.75  # type: ignore[list-item]
    assert rr._mean([]) is None
    assert rr._mean([float("nan")]) is None
    assert rr._mean([1 / 3]) == 0.3333


def _rows(n, **kw):
    base = {"llm_context_precision_with_reference": 1, "context_recall": 1, "faithfulness": 1,
            "answer_relevancy": 1, "answer_correctness": 1}
    base.update(kw)
    return [dict(base) for _ in range(n)]


def test_report_targets_met_boundaries_are_inclusive():
    s = [rr.EvalSample("q", "a", ["c"], "g", "semantic_risk")]
    r = rr.build_report(s, _rows(1, faithfulness=0.90, answer_correctness=0.7499),
                        pipeline="p", mode="smoke")
    assert r.targets_met["faithfulness"] is True
    assert r.targets_met["answer_correctness"] is False


def test_report_metric_without_any_scores_is_none_and_target_unknown():
    s = [rr.EvalSample("q", "a", ["c"], "g", "semantic_risk")]
    row = {"faithfulness": float("nan")}
    r = rr.build_report(s, [row], pipeline="p", mode="full")
    assert r.metrics["faithfulness"] is None and r.targets_met["faithfulness"] is None
    assert r.mode == "full" and r.num_samples == 1 and math.isfinite(len(r.per_sample))


def test_report_groups_by_query_type_and_keeps_per_sample_order():
    samples = [rr.EvalSample(f"q{i}", "a", ["c"], "g", qt)
               for i, qt in enumerate(["factual_numeric", "factual_numeric", "cross_reference"])]
    rows = _rows(3)
    rows[0]["faithfulness"], rows[1]["faithfulness"] = 1.0, 0.5
    r = rr.build_report(samples, rows, pipeline="p", mode="smoke")
    assert r.by_query_type["factual_numeric"]["faithfulness"] == 0.75
    assert r.by_query_type["cross_reference"]["faithfulness"] == 1.0
    assert [p["question"] for p in r.per_sample] == ["q0", "q1", "q2"]


def test_report_to_dict_has_iso_timestamp():
    r = rr.build_report([rr.EvalSample("q", "a", [], "g")], _rows(1), pipeline="p", mode="smoke")
    d = r.to_dict()
    assert {"pipeline", "mode", "timestamp", "num_samples", "metrics", "targets",
            "targets_met", "by_query_type", "per_sample"} <= set(d)
    from datetime import datetime
    datetime.fromisoformat(d["timestamp"])


def test_evaluate_samples_rejects_empty_input():
    with pytest.raises(ValueError):
        rr.evaluate_samples([])


async def test_generate_samples_reports_progress_and_copies_contexts():
    from amorphnet.evaluation.dataset_builder import get_eval_subset

    items = get_eval_subset("smoke")[:3]
    progress = []
    ctx = ["a", "b"]

    async def pipe(q, f):
        return rr.PipelineOutput(answer="x", contexts=ctx)

    samples = await rr.generate_samples(pipe, items, on_progress=lambda i, n, it: progress.append((i, n)))
    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert samples[0].contexts == ctx and samples[0].contexts is not ctx
    assert [s.query_type for s in samples] == [i.query_type for i in items]


async def test_aevaluate_samples_runs_in_worker_thread(monkeypatch):
    seen = {}

    def fake(samples, **kw):
        import threading
        seen["thread"] = threading.current_thread().name
        seen["kw"] = kw
        return "report"

    monkeypatch.setattr(rr, "evaluate_samples", fake)
    import threading
    assert await rr.aevaluate_samples([1], pipeline="p", mode="smoke") == "report"
    assert seen["thread"] != threading.current_thread().name
    assert seen["kw"] == {"pipeline": "p", "mode": "smoke"}
