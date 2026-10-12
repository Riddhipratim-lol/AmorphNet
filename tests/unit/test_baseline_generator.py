"""Phase 2.4 — ``amorphnet/generation/baseline_generator.py``.

Contract:
  ``async answer_question(question, filters, *, retrieve_fn=None, genai_client=None, limiter=None)
  -> {"answer": str, "contexts": list[str]}`` (shape consumed by ``scripts/run_evaluation.py``).
  ``build_baseline_prompt(question, contexts) -> str`` is naive concatenation, no ``[SOURCE`` envelopes.
  Gemini call is plain text: no ``response_schema``, no JSON mime type.
"""

from __future__ import annotations

from tests.support import FakeGenai, require

generator = require("amorphnet.generation.baseline_generator", "answer_question", "build_baseline_prompt")


def test_baseline_prompt_is_naive_concatenation_without_citation_envelopes():
    prompt = generator.build_baseline_prompt("What was revenue?", ["ctx one", "ctx two"])
    assert "What was revenue?" in prompt and "ctx one" in prompt and "ctx two" in prompt
    assert "[SOURCE" not in prompt


async def test_answer_question_returns_plain_text_and_contexts():
    async def fake_retrieve(question, **kw):
        return [{"text": "Revenue was $5.", "score": 0.9}, {"text": "Costs were $2.", "score": 0.8}]

    g = FakeGenai("  Revenue was $5.  ")
    out = await generator.answer_question("Revenue?", {"fiscal_year": 2025}, retrieve_fn=fake_retrieve,
                                          genai_client=g)
    assert out == {"answer": "Revenue was $5.", "contexts": ["Revenue was $5.", "Costs were $2."]}
    call = g.calls[0]
    assert call["model"] == "gemini-3.5-flash-lite"
    cfg = call["config"]
    assert getattr(cfg, "response_schema", None) is None
    assert getattr(cfg, "response_mime_type", None) in (None, "text/plain")
    assert "Revenue?" in str(call["contents"]) and "Costs were $2." in str(call["contents"])


async def test_answer_question_forwards_filters_to_retriever_and_uses_limiter(limiter):
    seen = {}

    async def fake_retrieve(question, **kw):
        seen.update(kw)
        return []

    g = FakeGenai("none")
    await generator.answer_question("q?", {"fiscal_year": 2025}, retrieve_fn=fake_retrieve,
                                    genai_client=g, limiter=limiter)
    assert seen.get("filters") == {"fiscal_year": 2025}
    assert limiter.acquired == 1
