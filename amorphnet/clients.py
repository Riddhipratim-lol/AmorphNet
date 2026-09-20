"""Shared cloud client singletons, connection managers, and rate-limiting guardrails.

All datastores and AI services run exclusively on managed cloud infrastructure.
Clients are lazily initialized singletons with token-bucket rate limiters and
retry wrappers to respect provider quotas and ceilings.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from amorphnet.config import Settings, get_settings

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Rate Limiters (Token-bucket guardrails)
# -----------------------------------------------------------------------------
try:
    from aiolimiter import AsyncLimiter

    _settings = get_settings()
    # Outbound Gemini rate limiter (14 RPM Free tier default; scales in Tier 1+)
    gemini_limiter = AsyncLimiter(max_rate=_settings.GEMINI_RPM_LIMIT, time_period=60)
    # Voyage AI embedding / rerank limiter (50 RPM default guardrail)
    voyage_limiter = AsyncLimiter(max_rate=_settings.VOYAGE_RPM_LIMIT, time_period=60)
except ImportError:
    # Dummy async limiter for environments without aiolimiter installed yet
    class DummyLimiter:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "DummyLimiter":
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

    gemini_limiter = DummyLimiter()  # type: ignore[assignment]
    voyage_limiter = DummyLimiter()  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# Singleton Holders
# -----------------------------------------------------------------------------
_voyage_client: Optional[Any] = None
_genai_client: Optional[Any] = None
_qdrant_client: Optional[Any] = None
_neo4j_driver: Optional[Any] = None
_redis_client: Optional[Any] = None


def get_voyage_client(settings: Optional[Settings] = None) -> Any:
    """Return cached Voyage AI AsyncClient singleton with retry backoff."""
    global _voyage_client
    if _voyage_client is None:
        import voyageai

        cfg = settings or get_settings()
        if not cfg.VOYAGE_API_KEY:
            raise ValueError("VOYAGE_API_KEY is not configured in settings or environment.")
        _voyage_client = voyageai.AsyncClient(
            api_key=cfg.VOYAGE_API_KEY,
            max_retries=cfg.VOYAGE_MAX_RETRIES,
            timeout=30.0,
        )
    return _voyage_client


def get_genai_client(settings: Optional[Settings] = None) -> Any:
    """Return cached Google GenAI Client singleton for Gemini 3.5 Flash-Lite."""
    global _genai_client
    if _genai_client is None:
        from google import genai

        cfg = settings or get_settings()
        if not cfg.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not configured in settings or environment.")
        _genai_client = genai.Client(api_key=cfg.GOOGLE_API_KEY)
    return _genai_client


def get_qdrant_client(settings: Optional[Settings] = None) -> Any:
    """Return cached Qdrant Cloud AsyncQdrantClient singleton."""
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import AsyncQdrantClient

        cfg = settings or get_settings()
        if not cfg.QDRANT_URL or not cfg.QDRANT_API_KEY:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be configured in settings.")
        _qdrant_client = AsyncQdrantClient(
            url=cfg.QDRANT_URL,
            api_key=cfg.QDRANT_API_KEY,
            timeout=30.0,
        )
    return _qdrant_client


def get_neo4j_driver(settings: Optional[Settings] = None) -> Any:
    """Return cached Neo4j AuraDB AsyncGraphDatabase driver singleton."""
    global _neo4j_driver
    if _neo4j_driver is None:
        from neo4j import AsyncGraphDatabase

        cfg = settings or get_settings()
        if not cfg.NEO4J_URI or not cfg.NEO4J_PASSWORD:
            raise ValueError("NEO4J_URI and NEO4J_PASSWORD must be configured in settings.")
        _neo4j_driver = AsyncGraphDatabase.driver(
            cfg.NEO4J_URI,
            auth=(cfg.NEO4J_USERNAME, cfg.NEO4J_PASSWORD),
        )
    return _neo4j_driver


def get_redis_client(settings: Optional[Settings] = None) -> Any:
    """Return cached Upstash Redis or redis.asyncio client singleton."""
    global _redis_client
    if _redis_client is None:
        cfg = settings or get_settings()
        if cfg.UPSTASH_REDIS_REST_URL and cfg.UPSTASH_REDIS_REST_TOKEN:
            try:
                from upstash_redis.asyncio import Redis as UpstashRedis

                _redis_client = UpstashRedis(
                    url=cfg.UPSTASH_REDIS_REST_URL,
                    token=cfg.UPSTASH_REDIS_REST_TOKEN,
                )
            except ImportError:
                import redis.asyncio as aioredis

                _redis_client = aioredis.from_url(
                    cfg.REDIS_URL or cfg.UPSTASH_REDIS_REST_URL,
                    decode_responses=True,
                )
        elif cfg.REDIS_URL:
            import redis.asyncio as aioredis

            _redis_client = aioredis.from_url(cfg.REDIS_URL, decode_responses=True)
        else:
            raise ValueError(
                "Either (UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN) or REDIS_URL must be set."
            )
    return _redis_client


async def close_all_clients() -> None:
    """Gracefully close all open client connection pools."""
    global _qdrant_client, _neo4j_driver, _redis_client

    if _qdrant_client is not None:
        try:
            await _qdrant_client.close()
            logger.info("Closed Qdrant Cloud client.")
        except Exception as err:
            logger.warning("Error closing Qdrant client: %s", err)
        _qdrant_client = None

    if _neo4j_driver is not None:
        try:
            await _neo4j_driver.close()
            logger.info("Closed Neo4j AuraDB driver.")
        except Exception as err:
            logger.warning("Error closing Neo4j driver: %s", err)
        _neo4j_driver = None

    if _redis_client is not None:
        try:
            if hasattr(_redis_client, "close"):
                await _redis_client.close()
            logger.info("Closed Redis connection.")
        except Exception as err:
            logger.warning("Error closing Redis client: %s", err)
        _redis_client = None
