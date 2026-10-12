"""Global pytest fixtures for AmorphNet.  See ``tests/support.py`` for the testing philosophy."""

from __future__ import annotations

import asyncio
import itertools
import os

import pytest

from tests.support import (
    CountingLimiter,
    FakeGenai,
    FakeNeo4j,
    FakeQdrant,
    FakeRedis,
    FakeVoyage,
)

_SERVICE_ENV_KEYS = (
    "LLAMA_CLOUD_API_KEY", "VOYAGE_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION",
    "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GOOGLE_API_KEY", "UPSTASH_REDIS_REST_URL",
    "UPSTASH_REDIS_REST_TOKEN", "REDIS_URL", "LOG_LEVEL", "MAX_CONCURRENT_INGESTIONS",
    "QUERY_TIMEOUT_SECONDS", "CACHE_TTL_SECONDS", "LLAMAPARSE_MAX_CONCURRENCY",
    "VOYAGE_MAX_RETRIES", "VOYAGE_RPM_LIMIT", "GEMINI_RPM_LIMIT", "EVAL_MODE",
)


def pytest_collection_modifyitems(config, items):
    """``live`` tests only run with RUN_LIVE_TESTS=1 (they hit real, billable cloud APIs)."""
    if os.getenv("RUN_LIVE_TESTS") == "1":
        return
    skip_live = pytest.mark.skip(reason="set RUN_LIVE_TESTS=1 to run live cloud tests")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture
def clean_env(monkeypatch):
    """Remove every AmorphNet env var so ``Settings(_env_file=None)`` yields pure defaults."""
    for key in _SERVICE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


@pytest.fixture
def no_sleep(monkeypatch):
    """Make tenacity / asyncio back-off instantaneous; records requested delays."""
    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay, result=None):
        delays.append(delay)
        await real_sleep(0)
        return result

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return delays


@pytest.fixture
def limiter():
    return CountingLimiter()


@pytest.fixture
def voyage():
    return FakeVoyage()


@pytest.fixture
def qdrant():
    return FakeQdrant()


@pytest.fixture
def neo4j():
    return FakeNeo4j()


@pytest.fixture
def redis():
    return FakeRedis()


@pytest.fixture
def genai_factory():
    """``genai_factory(responder)`` -> FakeGenai."""
    return FakeGenai


@pytest.fixture
def make_chunk():
    """Factory for valid ``Chunk`` objects (skips when ``chunk_schema`` is still a stub)."""
    from tests.support import require

    schema = require("amorphnet.ingestion.chunk_schema", "Chunk")
    counter = itertools.count(1)

    def _make(**overrides):
        n = next(counter)
        data = dict(
            chunk_id=f"chunk-{n}",
            document_id="doc-1",
            company_name="Apple Inc.",
            ticker="AAPL",
            filing_type="10-K",
            fiscal_year=2024,
            fiscal_quarter="Annual",
            page_number=1,
            section_header="Item 8",
            element_type="text",
            chunk_index=0,
            token_count=5,
            char_offset_start=0,
            char_offset_end=12,
            text=f"Sample chunk text {n}.",
        )
        data.update(overrides)
        return schema.Chunk(**data)

    return _make


@pytest.fixture
def patch_clients(monkeypatch):
    """Route ``amorphnet.clients.get_*`` to fakes.

    Usage: ``patch_clients(voyage=..., genai=..., qdrant=..., neo4j=..., redis=...)``.
    """
    from amorphnet import clients

    def _patch(**fakes):
        mapping = {
            "voyage": "get_voyage_client",
            "genai": "get_genai_client",
            "qdrant": "get_qdrant_client",
            "neo4j": "get_neo4j_driver",
            "redis": "get_redis_client",
        }
        for key, fake in fakes.items():
            monkeypatch.setattr(clients, mapping[key], lambda *a, _f=fake, **k: _f)
        # Never throttle in tests.
        monkeypatch.setattr(clients, "gemini_limiter", CountingLimiter())
        monkeypatch.setattr(clients, "voyage_limiter", CountingLimiter())

    return _patch
