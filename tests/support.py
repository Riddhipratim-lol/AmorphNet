"""Shared test support: contract loader + in-memory fakes for every cloud dependency.

Testing philosophy
------------------
* **Contract-first (TDD).**  Most Tier 1/Tier 2 modules are still 5-line stubs.  Each test
  module declares the public API it expects through :func:`require`.  While a module is
  still a stub the whole test file is *skipped* (with the list of missing names); the moment
  the names exist the tests run for real and fail loudly on any contract violation.
* **No network, no cloud, no sleeping.**  Voyage, Gemini, Qdrant, Neo4j and Redis are
  replaced by the fakes below.  Implementations must therefore obtain clients through
  ``amorphnet.clients`` *at call time* (``from amorphnet import clients; clients.get_x()``)
  or accept them via keyword arguments, so tests can inject fakes.
* **Fake call-shape contracts** (implementations must use exactly these call shapes):

  - Voyage:  ``await client.embed(texts=[...], model=..., input_type=...)`` ->
    ``.embeddings``; ``await client.rerank(query=..., documents=[...], model=..., top_k=...)``
    -> ``.results[i].index / .relevance_score``.
  - Gemini:  ``await client.aio.models.generate_content(model=..., contents=..., config=...)``
    -> ``.text`` (JSON string when a ``response_schema`` is configured).
  - Qdrant:  ``await client.collection_exists / create_collection / create_payload_index /
    upsert / query_points / retrieve`` with keyword arguments.
  - Neo4j:   ``async with driver.session() as s: result = await s.run(cypher, params);
    rows = await result.data()``.
  - Redis:   ``get / set(key, value, ex=ttl) / hset(name, mapping=...) / hgetall / sadd /
    srem / scard / delete / ping`` (all awaitable).
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import math
from types import SimpleNamespace
from typing import Any, Callable

import pytest


# -----------------------------------------------------------------------------
# Contract loader
# -----------------------------------------------------------------------------
def require(module_name: str, *names: str) -> Any:
    """Import ``module_name`` and make sure it exposes ``names``; otherwise skip the module.

    Call at module import time in a test file::

        chunker = require("amorphnet.ingestion.chunker", "chunk_page", "split_text")
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as err:  # pragma: no cover - depends on implementation progress
        pytest.skip(f"{module_name} not importable yet: {err}", allow_module_level=True)
    missing = [n for n in names if not hasattr(module, n)]
    if missing:
        pytest.skip(
            f"{module_name} not implemented yet (missing: {', '.join(missing)})",
            allow_module_level=True,
        )
    return module


# -----------------------------------------------------------------------------
# Token counting
# -----------------------------------------------------------------------------
def word_count(text: str) -> int:
    """Deterministic whitespace 'tokenizer' injected as ``count_tokens`` in chunker tests."""
    return len(text.split())


# -----------------------------------------------------------------------------
# Limiter
# -----------------------------------------------------------------------------
class CountingLimiter:
    """Async-context-manager stand-in for ``aiolimiter.AsyncLimiter`` that counts acquisitions."""

    def __init__(self) -> None:
        self.acquired = 0

    async def __aenter__(self) -> "CountingLimiter":
        self.acquired += 1
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


# -----------------------------------------------------------------------------
# Voyage
# -----------------------------------------------------------------------------
def fake_vector(text: str, dim: int = 1024) -> list[float]:
    """Deterministic unit vector derived from ``text`` (same text -> same vector)."""
    seed = hashlib.sha256(text.encode()).digest()
    raw = [seed[i % len(seed)] / 255.0 - 0.5 for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [v / norm for v in raw]


class FakeVoyage:
    """Fake ``voyageai.AsyncClient`` recording every embed/rerank call."""

    def __init__(self, dim: int = 1024, rerank_scores: Callable[[str, str], float] | None = None):
        self.dim = dim
        self.embed_calls: list[dict[str, Any]] = []
        self.rerank_calls: list[dict[str, Any]] = []
        self.fail_rerank: Exception | None = None
        self._rerank_scores = rerank_scores or self._overlap

    @staticmethod
    def _overlap(query: str, doc: str) -> float:
        q, d = set(query.lower().split()), set(doc.lower().split())
        return len(q & d) / (len(q) or 1)

    async def embed(self, texts: list[str], model: str, input_type: str | None = None, **kw: Any):
        self.embed_calls.append({"texts": list(texts), "model": model, "input_type": input_type, **kw})
        return SimpleNamespace(
            embeddings=[fake_vector(t, self.dim) for t in texts],
            total_tokens=sum(len(t.split()) for t in texts),
        )

    async def rerank(self, query: str, documents: list[str], model: str, top_k: int | None = None, **kw):
        self.rerank_calls.append({"query": query, "documents": list(documents), "model": model,
                                  "top_k": top_k, **kw})
        if self.fail_rerank:
            raise self.fail_rerank
        scored = [
            SimpleNamespace(index=i, document=d, relevance_score=self._rerank_scores(query, d))
            for i, d in enumerate(documents)
        ]
        scored.sort(key=lambda r: r.relevance_score, reverse=True)
        if top_k:
            scored = scored[:top_k]
        return SimpleNamespace(results=scored, total_tokens=0)


# -----------------------------------------------------------------------------
# Gemini
# -----------------------------------------------------------------------------
def genai_error(code: int = 429):
    """Build a real ``google.genai.errors.APIError`` with the given HTTP code."""
    from google.genai.errors import APIError

    status = "RESOURCE_EXHAUSTED" if code == 429 else "INTERNAL"
    return APIError(code, {"error": {"message": "boom", "status": status, "code": code}})


class FakeGenai:
    """Fake ``google.genai.Client``.

    ``responder(call: dict) -> str | Exception | list`` is invoked per call.  A list is
    consumed one item per call (an ``Exception`` item is raised); a plain ``str`` is returned
    as ``response.text``.  ``calls`` records ``model``, ``contents`` and ``config``.
    """

    def __init__(self, responder: Callable[[dict[str, Any]], Any] | list | str | None = None):
        self.calls: list[dict[str, Any]] = []
        self._responder = responder
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._generate))

    async def _generate(self, model: str, contents: Any, config: Any = None, **kw: Any):
        call = {"model": model, "contents": contents, "config": config, **kw}
        self.calls.append(call)
        r = self._responder
        if callable(r):
            r = r(call)
        elif isinstance(r, list):
            r = r.pop(0)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(text=r if r is not None else "{}")

    # helpers ---------------------------------------------------------------
    @staticmethod
    def schema_of(call: dict[str, Any]) -> Any:
        return getattr(call["config"], "response_schema", None)


# -----------------------------------------------------------------------------
# Qdrant
# -----------------------------------------------------------------------------
class FakeQdrant:
    """Fake ``AsyncQdrantClient`` capturing calls; ``query_points`` serves ``points``."""

    def __init__(self, points: list[Any] | None = None, exists: bool = False):
        self.points = points or []
        self.exists = exists
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.stored: dict[str, Any] = {}
        self.in_flight = 0
        self.max_in_flight = 0
        self.upsert_delay = 0.0

    def of(self, name: str) -> list[dict[str, Any]]:
        return [kw for n, kw in self.calls if n == name]

    async def collection_exists(self, collection_name: str, **kw: Any) -> bool:
        self.calls.append(("collection_exists", {"collection_name": collection_name, **kw}))
        return self.exists

    async def create_collection(self, **kw: Any) -> bool:
        self.calls.append(("create_collection", kw))
        return True

    async def create_payload_index(self, **kw: Any) -> Any:
        self.calls.append(("create_payload_index", kw))
        return SimpleNamespace(status="ok")

    async def upsert(self, **kw: Any) -> Any:
        self.calls.append(("upsert", kw))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.upsert_delay)
            for p in kw.get("points", []):
                self.stored[str(p.id)] = p
        finally:
            self.in_flight -= 1
        return SimpleNamespace(status="completed")

    async def query_points(self, **kw: Any) -> Any:
        self.calls.append(("query_points", kw))
        return SimpleNamespace(points=list(self.points))

    async def retrieve(self, collection_name: str, ids: list[str], **kw: Any) -> list[Any]:
        self.calls.append(("retrieve", {"collection_name": collection_name, "ids": list(ids), **kw}))
        return [p for p in self.points if str(p.id) in {str(i) for i in ids}]


def scored_point(chunk_id: str, payload: dict[str, Any], score: float = 0.5) -> Any:
    """Shape of a Qdrant ``ScoredPoint`` / ``Record`` as consumed by retrievers."""
    return SimpleNamespace(id=chunk_id, score=score, payload=payload, vector=None)


# -----------------------------------------------------------------------------
# Neo4j
# -----------------------------------------------------------------------------
class _FakeResult:
    def __init__(self, rows: list[dict[str, Any]]):
        self._rows = rows

    async def data(self) -> list[dict[str, Any]]:
        return self._rows

    async def single(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None


class _FakeSession:
    def __init__(self, driver: "FakeNeo4j"):
        self._driver = driver

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def run(self, query: str, parameters: dict[str, Any] | None = None, **kw: Any):
        params = dict(parameters or {}, **kw)
        self._driver.queries.append((query, params))
        return _FakeResult(self._driver.responder(query, params))


class FakeNeo4j:
    """Fake ``AsyncGraphDatabase.driver`` recording Cypher + params."""

    def __init__(self, responder: Callable[[str, dict[str, Any]], list[dict[str, Any]]] | None = None):
        self.responder = responder or (lambda q, p: [])
        self.queries: list[tuple[str, dict[str, Any]]] = []

    def session(self, **kw: Any) -> _FakeSession:
        return _FakeSession(self)

    def cypher(self) -> str:
        """All Cypher text executed so far, joined (for substring assertions)."""
        return "\n".join(q for q, _ in self.queries)


# -----------------------------------------------------------------------------
# Redis
# -----------------------------------------------------------------------------
class FakeRedis:
    """Minimal async Redis (strings, hashes, sets) with TTL bookkeeping."""

    def __init__(self) -> None:
        self.kv: dict[str, Any] = {}
        self.ttls: dict[str, int] = {}
        self.hashes: dict[str, dict[str, Any]] = {}
        self.sets: dict[str, set[str]] = {}
        self.fail = False

    def _check(self) -> None:
        if self.fail:
            raise ConnectionError("redis unavailable")

    async def ping(self) -> bool:
        self._check()
        return True

    async def get(self, key: str) -> Any:
        self._check()
        return self.kv.get(key)

    async def set(self, key: str, value: Any, ex: int | None = None, **kw: Any) -> bool:
        self._check()
        self.kv[key] = value
        if ex is not None:
            self.ttls[key] = ex
        return True

    async def delete(self, *keys: str) -> int:
        self._check()
        return sum(1 for k in keys if self.kv.pop(k, None) is not None)

    async def hset(self, name: str, key: str | None = None, value: Any = None,
                   mapping: dict[str, Any] | None = None) -> int:
        self._check()
        h = self.hashes.setdefault(name, {})
        if key is not None:
            h[key] = value
        h.update(mapping or {})
        return len(h)

    async def hgetall(self, name: str) -> dict[str, Any]:
        self._check()
        return dict(self.hashes.get(name, {}))

    async def sadd(self, name: str, *values: str) -> int:
        self._check()
        s = self.sets.setdefault(name, set())
        before = len(s)
        s.update(values)
        return len(s) - before

    async def srem(self, name: str, *values: str) -> int:
        self._check()
        s = self.sets.setdefault(name, set())
        before = len(s)
        s.difference_update(values)
        return before - len(s)

    async def scard(self, name: str) -> int:
        self._check()
        return len(self.sets.get(name, set()))

    async def expire(self, name: str, seconds: int) -> bool:
        self._check()
        return True


# -----------------------------------------------------------------------------
# PDF builders (PyMuPDF)
# -----------------------------------------------------------------------------
def make_text_pdf(pages: list[str] | int = 3, *, title: str = "", author: str = "") -> bytes:
    """Build a real text PDF.  ``pages`` is a list of page texts or a page count."""
    import pymupdf

    texts = pages if isinstance(pages, list) else [f"Page {i + 1} body text." for i in range(pages)]
    doc = pymupdf.open()
    for text in texts:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    if title or author:
        doc.set_metadata({"title": title, "author": author})
    data = doc.tobytes()
    doc.close()
    return data


def make_scanned_pdf(pages: int = 3) -> bytes:
    """Build an image-only PDF (no extractable text) to emulate a scanned filing."""
    import pymupdf

    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), False)
    pix.clear_with(200)
    for _ in range(pages):
        page = doc.new_page()
        page.insert_image(pymupdf.Rect(72, 72, 272, 272), pixmap=pix)
    data = doc.tobytes()
    doc.close()
    return data


def make_encrypted_pdf(password: str = "secret") -> bytes:
    import pymupdf

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "classified")
    data = doc.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw=password, user_pw=password
    )
    doc.close()
    return data


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)
