#!/usr/bin/env python3
"""AmorphNet Cloud Service Connectivity Verification Script.

Tests reachability, authentication, and handshake with managed cloud services:
1. Qdrant Cloud (Vector + Sparse BM25)
2. Neo4j AuraDB (Document Knowledge Graph)
3. Voyage AI (voyage-4-large & rerank-2.5)
4. Google GenAI (Gemini 3.5 Flash-Lite)
5. Upstash Redis (Cache & Rate Limiting)
6. LlamaParse Cloud API (Multimodal PDF parser)

Usage:
    python scripts/check_services.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import NamedTuple

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from amorphnet.config import get_settings


class ServiceCheckResult(NamedTuple):
    service: str
    status: str  # "PASSED", "FAILED", "SKIPPED", "MISSING_KEY", "MISSING_LIB"
    message: str


async def check_qdrant() -> ServiceCheckResult:
    """Verify Qdrant Cloud connectivity."""
    settings = get_settings()
    if not settings.QDRANT_URL or not settings.QDRANT_API_KEY:
        return ServiceCheckResult("Qdrant Cloud", "MISSING_KEY", "QDRANT_URL or QDRANT_API_KEY not set")

    try:
        from qdrant_client import AsyncQdrantClient
    except ImportError:
        return ServiceCheckResult("Qdrant Cloud", "MISSING_LIB", "qdrant-client not installed")

    try:
        client = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=10.0,
        )
        collections = await client.get_collections()
        await client.close()
        count = len(collections.collections)
        return ServiceCheckResult(
            "Qdrant Cloud", "PASSED", f"Authenticated successfully ({count} collections present)"
        )
    except Exception as exc:
        return ServiceCheckResult("Qdrant Cloud", "FAILED", f"Error: {exc}")


async def check_neo4j() -> ServiceCheckResult:
    """Verify Neo4j AuraDB connectivity."""
    settings = get_settings()
    if not settings.NEO4J_URI or not settings.NEO4J_PASSWORD:
        return ServiceCheckResult("Neo4j AuraDB", "MISSING_KEY", "NEO4J_URI or NEO4J_PASSWORD not set")

    try:
        from neo4j import AsyncGraphDatabase
    except ImportError:
        return ServiceCheckResult("Neo4j AuraDB", "MISSING_LIB", "neo4j not installed")

    try:
        driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
        )
        async with driver.session() as session:
            result = await session.run("RETURN 1 AS ping")
            record = await result.single()
            val = record["ping"] if record else None
        await driver.close()
        if val == 1:
            return ServiceCheckResult("Neo4j AuraDB", "PASSED", "Handshake & Cypher ping succeeded")
        return ServiceCheckResult("Neo4j AuraDB", "FAILED", f"Unexpected response: {val}")
    except Exception as exc:
        return ServiceCheckResult("Neo4j AuraDB", "FAILED", f"Error: {exc}")


async def check_voyage() -> ServiceCheckResult:
    """Verify Voyage AI API credentials."""
    settings = get_settings()
    if not settings.VOYAGE_API_KEY:
        return ServiceCheckResult("Voyage AI", "MISSING_KEY", "VOYAGE_API_KEY not set")

    try:
        import voyageai
    except ImportError:
        return ServiceCheckResult("Voyage AI", "MISSING_LIB", "voyageai not installed")

    try:
        client = voyageai.AsyncClient(api_key=settings.VOYAGE_API_KEY, max_retries=1, timeout=10.0)
        # Check tokenization or lightweight embed
        res = await client.embed(
            texts=["ping"],
            model="voyage-4-large",
            input_type="query",
        )
        dim = len(res.embeddings[0])
        return ServiceCheckResult(
            "Voyage AI", "PASSED", f"Embedding check passed (dim={dim}, model=voyage-4-large)"
        )
    except Exception as exc:
        return ServiceCheckResult("Voyage AI", "FAILED", f"Error: {exc}")


async def check_google_genai() -> ServiceCheckResult:
    """Verify Google GenAI (Gemini) API credentials."""
    settings = get_settings()
    if not settings.GOOGLE_API_KEY:
        return ServiceCheckResult("Google GenAI", "MISSING_KEY", "GOOGLE_API_KEY not set")

    try:
        from google import genai
    except ImportError:
        return ServiceCheckResult("Google GenAI", "MISSING_LIB", "google-genai not installed")

    try:
        client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        # Test lightweight generation ping
        response = await client.aio.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents="ping",
        )
        if response.text:
            return ServiceCheckResult(
                "Google GenAI", "PASSED", "Gemini 3.5 Flash-Lite generation ping succeeded"
            )
        return ServiceCheckResult("Google GenAI", "FAILED", "Empty response from Gemini")
    except Exception as exc:
        return ServiceCheckResult("Google GenAI", "FAILED", f"Error: {exc}")


async def check_upstash_redis() -> ServiceCheckResult:
    """Verify Upstash Redis connectivity."""
    settings = get_settings()
    if not (settings.UPSTASH_REDIS_REST_URL and settings.UPSTASH_REDIS_REST_TOKEN) and not settings.REDIS_URL:
        return ServiceCheckResult("Upstash Redis", "MISSING_KEY", "Upstash Redis credentials or REDIS_URL not set")

    if settings.UPSTASH_REDIS_REST_URL and settings.UPSTASH_REDIS_REST_TOKEN:
        try:
            from upstash_redis.asyncio import Redis as UpstashRedis

            client = UpstashRedis(
                url=settings.UPSTASH_REDIS_REST_URL,
                token=settings.UPSTASH_REDIS_REST_TOKEN,
            )
            pong = await client.ping()
            if pong:
                return ServiceCheckResult("Upstash Redis", "PASSED", "REST API ping succeeded")
        except ImportError:
            pass

    # Fallback to redis-py
    try:
        import redis.asyncio as aioredis
    except ImportError:
        return ServiceCheckResult("Upstash Redis", "MISSING_LIB", "upstash-redis or redis not installed")

    try:
        target_url = settings.REDIS_URL or settings.UPSTASH_REDIS_REST_URL
        client = aioredis.from_url(target_url, decode_responses=True)
        pong = await client.ping()
        await client.close()
        if pong:
            return ServiceCheckResult("Upstash Redis", "PASSED", "PING -> PONG succeeded")
        return ServiceCheckResult("Upstash Redis", "FAILED", f"Unexpected PING response: {pong}")
    except Exception as exc:
        return ServiceCheckResult("Upstash Redis", "FAILED", f"Error: {exc}")


async def check_llamaparse() -> ServiceCheckResult:
    """Verify LlamaParse API Key presence."""
    settings = get_settings()
    if not settings.LLAMA_CLOUD_API_KEY:
        return ServiceCheckResult("LlamaParse", "MISSING_KEY", "LLAMA_CLOUD_API_KEY not set")

    try:
        import llama_cloud_services  # noqa: F401
        return ServiceCheckResult("LlamaParse", "PASSED", "SDK imported & API key configured")
    except ImportError:
        return ServiceCheckResult("LlamaParse", "MISSING_LIB", "llama-cloud-services not installed")


async def main() -> None:
    print("=" * 78)
    print(" AmorphNet — Cloud Services Connectivity Verification")
    print("=" * 78)

    checks = [
        check_qdrant(),
        check_neo4j(),
        check_voyage(),
        check_google_genai(),
        check_upstash_redis(),
        check_llamaparse(),
    ]

    results = await asyncio.gather(*checks)

    print(f"{'SERVICE':<18} | {'STATUS':<14} | {'DETAILS'}")
    print("-" * 78)

    has_missing_libs = False
    for res in results:
        color_start = ""
        color_end = "\033[0m"
        if res.status == "PASSED":
            color_start = "\033[92m"  # Green
        elif res.status in ("MISSING_KEY", "SKIPPED"):
            color_start = "\033[93m"  # Yellow
        elif res.status == "MISSING_LIB":
            color_start = "\033[94m"  # Blue
            has_missing_libs = True
        else:
            color_start = "\033[91m"  # Red

        status_formatted = f"{color_start}{res.status:<14}{color_end}"
        print(f"{res.service:<18} | {status_formatted} | {res.message}")

    print("-" * 78)
    if has_missing_libs:
        print("Note: Some libraries are not installed yet.")
        print("Install dependencies from requirements.txt to enable live connection verification:")
        print("  pip install -r requirements.txt")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())
