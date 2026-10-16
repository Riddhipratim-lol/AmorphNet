"""Phase 7.1 / 7.2 — ``amorphnet/ingestion/qdrant_indexer.py`` (collection, embedding, upsert).

Contract:
  Constants: ``DENSE_VECTOR_NAME="dense"``, ``SPARSE_VECTOR_NAME="sparse"``, ``DENSE_DIM=1024``,
    ``EMBEDDING_MODEL="voyage-4-large"``, ``BM25_MODEL="Qdrant/bm25"``, ``EMBED_BATCH_SIZE=128``,
    ``UPSERT_BATCH_SIZE=100``, ``UPSERT_CONCURRENCY=5``,
    ``PAYLOAD_INDEXES == [("company_name","keyword"), ("ticker","keyword"), ("filing_type","keyword"),
      ("fiscal_year","integer"), ("fiscal_quarter","keyword"), ("element_type","keyword"),
      ("page_number","integer"), ("section_header","text")]``
  ``async ensure_collection(client, collection_name=None)``
      ``create_collection(collection_name=..., vectors_config={"dense": VectorParams(1024, COSINE,
      hnsw m=16/ef_construct=200)}, sparse_vectors_config={"sparse": SparseVectorParams(modifier=IDF)})``
      unless ``collection_exists``; then ``create_payload_index(collection_name=, field_name=,
      field_schema=)`` for every entry of ``PAYLOAD_INDEXES`` (always — it is idempotent).
      Default name: ``get_settings().QDRANT_COLLECTION``.
  ``async embed_texts(texts, *, voyage_client, limiter=None, batch_size=128) -> list[list[float]]``
      ``voyage_client.embed(texts=..., model="voyage-4-large", input_type="document")`` per batch of
      <= 128 (``batch_size > 128`` => ``ValueError``); order preserved; ``limiter`` (default
      ``clients.voyage_limiter``) acquired once per call; empty input => ``[]`` with no calls.
  ``build_point(chunk, dense_vector) -> PointStruct`` / ``build_points(chunks, vectors)``
      ``id == chunk.chunk_id``; ``vector == {"dense": vec, "sparse": models.Document(text=chunk.text,
      model="Qdrant/bm25")}`` (server-side BM25); ``payload == chunk.to_payload()``;
      length mismatch => ``ValueError``.
  ``async index_chunks(chunks, *, qdrant_client, voyage_client, collection_name=None, limiter=None,
      upsert_batch_size=100, concurrency=5) -> int``
      Embeds chunk texts (<=128/call), upserts <=100 points/call with at most ``concurrency`` upserts
      in flight; returns the number of points written; errors propagate (no silent loss).
"""

from __future__ import annotations

import pytest
from qdrant_client import models

from amorphnet.config import get_settings
from tests.support import FakeQdrant, FakeVoyage, fake_vector, require

qi = require("amorphnet.ingestion.qdrant_indexer", "ensure_collection", "embed_texts", "build_point",
             "build_points", "index_chunks", "PAYLOAD_INDEXES", "DENSE_DIM", "EMBEDDING_MODEL",
             "BM25_MODEL", "EMBED_BATCH_SIZE", "UPSERT_BATCH_SIZE", "UPSERT_CONCURRENCY",
             "DENSE_VECTOR_NAME", "SPARSE_VECTOR_NAME")

COLLECTION = "financial_chunks"


@pytest.fixture
def chunks(make_chunk):
    return lambda n: [make_chunk(chunk_id=f"00000000-0000-0000-0000-{i:012d}", text=f"chunk text {i}")
                      for i in range(n)]


# ----------------------------------------------------------------------------- constants
def test_constants():
    assert (qi.DENSE_VECTOR_NAME, qi.SPARSE_VECTOR_NAME, qi.DENSE_DIM) == ("dense", "sparse", 1024)
    assert (qi.EMBEDDING_MODEL, qi.BM25_MODEL) == ("voyage-4-large", "Qdrant/bm25")
    assert (qi.EMBED_BATCH_SIZE, qi.UPSERT_BATCH_SIZE, qi.UPSERT_CONCURRENCY) == (128, 100, 5)


def test_payload_index_plan_matches_architecture():
    assert [tuple(x) for x in qi.PAYLOAD_INDEXES] == [
        ("company_name", "keyword"), ("ticker", "keyword"), ("filing_type", "keyword"),
        ("fiscal_year", "integer"), ("fiscal_quarter", "keyword"), ("element_type", "keyword"),
        ("page_number", "integer"), ("section_header", "text")]


# ----------------------------------------------------------------------------- 7.1 collection
async def test_ensure_collection_creates_named_dense_and_sparse_vectors():
    q = FakeQdrant(exists=False)
    await qi.ensure_collection(q, COLLECTION)
    (kw,) = q.of("create_collection")
    assert kw["collection_name"] == COLLECTION
    dense = kw["vectors_config"]["dense"]
    assert dense.size == 1024 and dense.distance == models.Distance.COSINE
    assert dense.hnsw_config.m == 16 and dense.hnsw_config.ef_construct == 200
    sparse = kw["sparse_vectors_config"]["sparse"]
    assert sparse.modifier == models.Modifier.IDF  # server-side BM25 IDF weighting
    assert set(kw["vectors_config"]) == {"dense"} and set(kw["sparse_vectors_config"]) == {"sparse"}


async def test_ensure_collection_creates_all_eight_payload_indexes():
    q = FakeQdrant(exists=False)
    await qi.ensure_collection(q, COLLECTION)
    idx = {kw["field_name"]: kw["field_schema"] for kw in q.of("create_payload_index")}
    assert len(q.of("create_payload_index")) == 8
    assert all(kw["collection_name"] == COLLECTION for kw in q.of("create_payload_index"))
    for f in ("company_name", "ticker", "filing_type", "fiscal_quarter", "element_type"):
        assert idx[f] == models.PayloadSchemaType.KEYWORD
    for f in ("fiscal_year", "page_number"):
        assert idx[f] == models.PayloadSchemaType.INTEGER
    assert "text" in str(idx["section_header"]).lower()  # full-text index


async def test_ensure_collection_is_idempotent_but_still_ensures_indexes():
    q = FakeQdrant(exists=True)
    await qi.ensure_collection(q, COLLECTION)
    assert q.of("create_collection") == []
    assert len(q.of("create_payload_index")) == 8


async def test_ensure_collection_default_name_from_settings():
    q = FakeQdrant(exists=False)
    await qi.ensure_collection(q)
    assert q.of("create_collection")[0]["collection_name"] == get_settings().QDRANT_COLLECTION


# ----------------------------------------------------------------------------- 7.2 embeddings
async def test_embed_texts_batches_of_128_in_order():
    v = FakeVoyage()
    texts = [f"t{i}" for i in range(300)]
    vecs = await qi.embed_texts(texts, voyage_client=v)
    assert [len(c["texts"]) for c in v.embed_calls] == [128, 128, 44]
    assert all(c["model"] == "voyage-4-large" and c["input_type"] == "document" for c in v.embed_calls)
    assert len(vecs) == 300 and len(vecs[0]) == 1024
    assert vecs[217] == fake_vector("t217")  # order preserved across batches


async def test_embed_texts_exact_boundary_and_limiter(limiter):
    v = FakeVoyage()
    await qi.embed_texts([f"t{i}" for i in range(128)], voyage_client=v, limiter=limiter)
    assert len(v.embed_calls) == 1 and limiter.acquired == 1


async def test_embed_texts_limiter_per_call(limiter):
    v = FakeVoyage()
    await qi.embed_texts([f"t{i}" for i in range(129)], voyage_client=v, limiter=limiter)
    assert limiter.acquired == len(v.embed_calls) == 2


async def test_embed_texts_empty_and_oversized_batch_size():
    v = FakeVoyage()
    assert await qi.embed_texts([], voyage_client=v) == [] and v.embed_calls == []
    with pytest.raises(ValueError):
        await qi.embed_texts(["a"], voyage_client=v, batch_size=129)


async def test_embed_texts_default_limiter_from_clients(patch_clients):
    from amorphnet import clients

    v = FakeVoyage()
    patch_clients(voyage=v)
    await qi.embed_texts(["a", "b"])
    assert clients.voyage_limiter.acquired == 1 and len(v.embed_calls) == 1


# ----------------------------------------------------------------------------- points
def test_build_point_has_named_dense_and_bm25_sparse_and_full_payload(make_chunk):
    c = make_chunk(chunk_id="11111111-1111-1111-1111-111111111111", element_type="table",
                   table_caption="Net sales", financial_metrics=["revenue"], breadcrumb="A > B")
    vec = fake_vector("x")
    p = qi.build_point(c, vec)
    assert str(p.id) == c.chunk_id
    assert p.vector["dense"] == vec
    sparse = p.vector["sparse"]
    assert sparse.text == c.text and sparse.model == "Qdrant/bm25"
    assert p.payload == c.to_payload()
    for k in ("company_name", "fiscal_year", "element_type", "section_header", "page_number",
              "financial_metrics", "breadcrumb", "table_caption", "text"):
        assert k in p.payload


def test_build_points_alignment_and_mismatch(chunks):
    cs = chunks(3)
    vecs = [fake_vector(c.text) for c in cs]
    pts = qi.build_points(cs, vecs)
    assert [str(p.id) for p in pts] == [c.chunk_id for c in cs]
    assert [p.vector["dense"] for p in pts] == vecs
    with pytest.raises(ValueError):
        qi.build_points(cs, vecs[:2])


# ----------------------------------------------------------------------------- indexing
async def test_index_chunks_upserts_in_batches_of_100(chunks):
    q, v = FakeQdrant(), FakeVoyage()
    n = await qi.index_chunks(chunks(250), qdrant_client=q, voyage_client=v, collection_name=COLLECTION)
    assert n == 250 and len(q.stored) == 250
    assert [len(kw["points"]) for kw in q.of("upsert")] == [100, 100, 50]
    assert all(kw["collection_name"] == COLLECTION for kw in q.of("upsert"))
    assert [len(c["texts"]) for c in v.embed_calls] == [128, 122]


async def test_index_chunks_embeds_chunk_text_and_stores_matching_vector(chunks):
    q, v = FakeQdrant(), FakeVoyage()
    cs = chunks(5)
    await qi.index_chunks(cs, qdrant_client=q, voyage_client=v, collection_name=COLLECTION)
    assert [t for c in v.embed_calls for t in c["texts"]] == [c.text for c in cs]
    stored = q.stored[cs[2].chunk_id]
    assert stored.vector["dense"] == fake_vector(cs[2].text) and stored.vector["sparse"].text == cs[2].text


async def test_upsert_concurrency_is_capped_at_five_but_actually_parallel(chunks):
    q, v = FakeQdrant(), FakeVoyage()
    q.upsert_delay = 0.02
    await qi.index_chunks(chunks(1200), qdrant_client=q, voyage_client=v, collection_name=COLLECTION)
    assert len(q.of("upsert")) == 12
    assert q.max_in_flight == 5


async def test_custom_concurrency_and_batch_size(chunks):
    q, v = FakeQdrant(), FakeVoyage()
    q.upsert_delay = 0.01
    await qi.index_chunks(chunks(60), qdrant_client=q, voyage_client=v, collection_name=COLLECTION,
                          upsert_batch_size=10, concurrency=2)
    assert len(q.of("upsert")) == 6 and q.max_in_flight == 2


async def test_reindexing_same_chunks_is_idempotent(chunks):
    q, v = FakeQdrant(), FakeVoyage()
    cs = chunks(30)
    await qi.index_chunks(cs, qdrant_client=q, voyage_client=v, collection_name=COLLECTION)
    await qi.index_chunks(cs, qdrant_client=q, voyage_client=v, collection_name=COLLECTION)
    assert len(q.stored) == 30  # same point ids overwrite instead of duplicating


async def test_empty_input_is_noop(chunks):
    q, v = FakeQdrant(), FakeVoyage()
    assert await qi.index_chunks([], qdrant_client=q, voyage_client=v, collection_name=COLLECTION) == 0
    assert q.calls == [] and v.embed_calls == []


async def test_upsert_failure_propagates(chunks):
    q, v = FakeQdrant(), FakeVoyage()

    async def boom(**kw):
        raise ConnectionError("qdrant down")

    q.upsert = boom
    with pytest.raises(ConnectionError):
        await qi.index_chunks(chunks(5), qdrant_client=q, voyage_client=v, collection_name=COLLECTION)


async def test_limiter_is_applied_to_embedding_calls(chunks, limiter):
    q, v = FakeQdrant(), FakeVoyage()
    await qi.index_chunks(chunks(200), qdrant_client=q, voyage_client=v, collection_name=COLLECTION,
                          limiter=limiter)
    assert limiter.acquired == len(v.embed_calls) == 2


async def test_default_clients_and_collection_name(chunks, patch_clients):
    q, v = FakeQdrant(), FakeVoyage()
    patch_clients(qdrant=q, voyage=v)
    await qi.index_chunks(chunks(3))
    assert q.of("upsert")[0]["collection_name"] == get_settings().QDRANT_COLLECTION
