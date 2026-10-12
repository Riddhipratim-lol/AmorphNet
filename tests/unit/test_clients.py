"""Phase 0.4 — client singleton factories, rate limiters and graceful shutdown."""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from amorphnet import clients
from amorphnet.config import Settings

RESET = ("_voyage_client", "_genai_client", "_qdrant_client", "_neo4j_driver", "_redis_client")


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    for name in RESET:
        monkeypatch.setattr(clients, name, None)


@pytest.fixture
def cfg(clean_env):
    return lambda **kw: Settings(_env_file=None, **kw)


def test_rate_limiters_match_provider_guardrails():
    assert clients.gemini_limiter.max_rate == clients.get_settings().GEMINI_RPM_LIMIT
    assert clients.gemini_limiter.time_period == 60
    assert clients.voyage_limiter.max_rate == clients.get_settings().VOYAGE_RPM_LIMIT
    assert clients.voyage_limiter.time_period == 60


@pytest.mark.parametrize(
    "getter",
    ["get_voyage_client", "get_genai_client", "get_qdrant_client", "get_neo4j_driver", "get_redis_client"],
)
def test_getters_fail_fast_without_credentials(cfg, getter):
    with pytest.raises(ValueError):
        getattr(clients, getter)(cfg())


def test_voyage_client_uses_retry_budget_and_is_singleton(cfg, monkeypatch):
    import voyageai

    created = []

    class Fake:
        def __init__(self, **kw):
            created.append(kw)

    monkeypatch.setattr(voyageai, "AsyncClient", Fake)
    a = clients.get_voyage_client(cfg(VOYAGE_API_KEY="pa-1"))
    b = clients.get_voyage_client(cfg(VOYAGE_API_KEY="pa-1"))
    assert a is b and len(created) == 1
    assert created[0]["api_key"] == "pa-1"
    assert created[0]["max_retries"] == 5


def test_qdrant_client_receives_url_and_key(cfg, monkeypatch):
    import qdrant_client

    created = []
    monkeypatch.setattr(qdrant_client, "AsyncQdrantClient", lambda **kw: created.append(kw) or object())
    clients.get_qdrant_client(cfg(QDRANT_URL="https://q", QDRANT_API_KEY="k"))
    assert created[0]["url"] == "https://q" and created[0]["api_key"] == "k"


def test_qdrant_requires_both_url_and_key(cfg):
    with pytest.raises(ValueError):
        clients.get_qdrant_client(cfg(QDRANT_URL="https://q"))


def test_neo4j_driver_uses_aura_uri_and_basic_auth(cfg, monkeypatch):
    import neo4j

    seen = {}
    monkeypatch.setattr(
        neo4j.AsyncGraphDatabase, "driver",
        staticmethod(lambda uri, auth=None, **kw: seen.update(uri=uri, auth=auth) or object()),
    )
    clients.get_neo4j_driver(cfg(NEO4J_URI="neo4j+s://x", NEO4J_USERNAME="neo4j", NEO4J_PASSWORD="pw"))
    assert seen == {"uri": "neo4j+s://x", "auth": ("neo4j", "pw")}


def test_genai_client_uses_api_key(cfg, monkeypatch):
    from google import genai

    seen = {}
    monkeypatch.setattr(genai, "Client", lambda **kw: seen.update(kw) or object())
    clients.get_genai_client(cfg(GOOGLE_API_KEY="AIza"))
    assert seen["api_key"] == "AIza"


def test_redis_prefers_upstash_rest_then_falls_back_to_redis_url(cfg, monkeypatch):
    fake_mod = types.ModuleType("upstash_redis.asyncio")
    seen = {}
    fake_mod.Redis = lambda **kw: seen.update(kw) or "upstash"
    monkeypatch.setitem(sys.modules, "upstash_redis.asyncio", fake_mod)
    got = clients.get_redis_client(cfg(UPSTASH_REDIS_REST_URL="https://u", UPSTASH_REDIS_REST_TOKEN="t"))
    assert got == "upstash" and seen == {"url": "https://u", "token": "t"}

    monkeypatch.setattr(clients, "_redis_client", None)
    import redis.asyncio as aioredis

    monkeypatch.setattr(aioredis, "from_url", lambda url, **kw: ("redis", url))
    assert clients.get_redis_client(cfg(REDIS_URL="rediss://h:1")) == ("redis", "rediss://h:1")


async def test_close_all_clients_closes_and_resets(monkeypatch):
    q, n, r = MagicMock(close=AsyncMock()), MagicMock(close=AsyncMock()), MagicMock(close=AsyncMock())
    monkeypatch.setattr(clients, "_qdrant_client", q)
    monkeypatch.setattr(clients, "_neo4j_driver", n)
    monkeypatch.setattr(clients, "_redis_client", r)
    await clients.close_all_clients()
    q.close.assert_awaited_once(); n.close.assert_awaited_once(); r.close.assert_awaited_once()
    assert clients._qdrant_client is None and clients._neo4j_driver is None and clients._redis_client is None


async def test_close_all_clients_survives_close_errors(monkeypatch):
    bad = MagicMock(close=AsyncMock(side_effect=RuntimeError("x")))
    monkeypatch.setattr(clients, "_qdrant_client", bad)
    await clients.close_all_clients()  # must not raise
    assert clients._qdrant_client is None


async def test_close_all_clients_is_noop_when_nothing_open():
    await clients.close_all_clients()
