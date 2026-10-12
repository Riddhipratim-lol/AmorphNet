"""Phase 2.2 — ``amorphnet/ingestion/baseline_indexer.py``.

Contract:
  ``BASELINE_COLLECTION == "financial_chunks_baseline"``
  ``async create_baseline_collection(client)`` — idempotent; dense-only 1024-dim Cosine.
  ``async index_baseline_chunks(chunks, *, qdrant_client, voyage_client, batch_size=128) -> int``
  embeds with ``input_type="document"`` in batches <= 128, upserts payload
  ``{document_id, page_number, chunk_id, text}`` with point id == chunk_id, no sparse vector.
"""

from __future__ import annotations

from tests.support import FakeQdrant, FakeVoyage, require

indexer = require("amorphnet.ingestion.baseline_indexer", "BASELINE_COLLECTION",
                  "create_baseline_collection", "index_baseline_chunks")


def _chunks(n):
    return [{"document_id": "d", "page_number": 1, "chunk_id": f"c{i}", "text": f"text {i}"}
            for i in range(n)]


def test_collection_name():
    assert indexer.BASELINE_COLLECTION == "financial_chunks_baseline"


def _dense_params(kw):
    vc = kw["vectors_config"]
    return vc["dense"] if isinstance(vc, dict) else vc


async def test_create_baseline_collection_dense_only():
    q = FakeQdrant(exists=False)
    await indexer.create_baseline_collection(q)
    kw = q.of("create_collection")[0]
    assert kw["collection_name"] == "financial_chunks_baseline"
    p = _dense_params(kw)
    assert p.size == 1024 and p.distance.value.lower() == "cosine"
    assert not kw.get("sparse_vectors_config")


async def test_create_baseline_collection_is_idempotent():
    q = FakeQdrant(exists=True)
    await indexer.create_baseline_collection(q)
    assert q.of("create_collection") == []


async def test_index_baseline_batches_embeddings_by_128_and_upserts_all():
    q, v = FakeQdrant(), FakeVoyage()
    n = await indexer.index_baseline_chunks(_chunks(300), qdrant_client=q, voyage_client=v)
    assert n == 300
    assert [len(c["texts"]) for c in v.embed_calls] == [128, 128, 44]
    assert all(c["model"] == "voyage-4-large" and c["input_type"] == "document" for c in v.embed_calls)
    assert len(q.stored) == 300
    p = q.stored["c7"]
    assert p.payload["text"] == "text 7" and p.payload["document_id"] == "d"
    assert all(u["collection_name"] == "financial_chunks_baseline" for u in q.of("upsert"))


async def test_index_baseline_has_no_sparse_vectors_and_empty_is_noop():
    q, v = FakeQdrant(), FakeVoyage()
    await indexer.index_baseline_chunks(_chunks(2), qdrant_client=q, voyage_client=v)
    vec = q.stored["c0"].vector
    assert not isinstance(vec, dict) or "sparse" not in vec
    q2, v2 = FakeQdrant(), FakeVoyage()
    assert await indexer.index_baseline_chunks([], qdrant_client=q2, voyage_client=v2) == 0
    assert q2.calls == [] and v2.embed_calls == []

