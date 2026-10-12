"""Phase 2.3 — ``amorphnet/retrieval/baseline_retriever.py``.

Contract:
  ``async retrieve(question, *, qdrant_client, voyage_client, top_k=6, filters=None) -> list[dict]``
  Query embedded with ``input_type="query"``; a single dense ``query_points`` on
  ``financial_chunks_baseline`` (no filter, no prefetch/BM25/HyDE); returns payload dicts + ``score``.
"""

from __future__ import annotations

import pytest

from tests.support import FakeQdrant, FakeVoyage, require, scored_point

retriever = require("amorphnet.retrieval.baseline_retriever", "retrieve")


async def test_retrieve_embeds_query_and_searches_top6_dense_only():
    pts = [scored_point(f"c{i}", {"text": f"t{i}", "page_number": i, "document_id": "d",
                                    "chunk_id": f"c{i}"}, 1 - i / 10) for i in range(6)]
    q, v = FakeQdrant(points=pts), FakeVoyage()
    out = await retriever.retrieve("net sales?", qdrant_client=q, voyage_client=v,
                                   filters={"fiscal_year": 2025})
    assert v.embed_calls[0]["input_type"] == "query" and v.embed_calls[0]["model"] == "voyage-4-large"
    kw = q.of("query_points")[0]
    assert kw["collection_name"] == "financial_chunks_baseline" and kw["limit"] == 6
    assert kw.get("query_filter") is None and not kw.get("prefetch")  # no filters/BM25/HyDE
    assert [r["text"] for r in out] == [f"t{i}" for i in range(6)]
    assert out[0]["score"] == pytest.approx(1.0)


async def test_retrieve_custom_top_k_and_empty_result():
    q, v = FakeQdrant(points=[]), FakeVoyage()
    assert await retriever.retrieve("x", qdrant_client=q, voyage_client=v, top_k=3) == []
    assert q.of("query_points")[0]["limit"] == 3

