"""Centralized configuration and environment settings for AmorphNet.

All datastores and AI services run exclusively on managed cloud APIs.
Settings are loaded from environment variables and optional .env file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

try:
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    # Graceful fallback before dependencies are installed
    class BaseSettings:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class SettingsConfigDict:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            pass

    def Field(default=None, **kwargs):  # type: ignore[no-redef]
        return default


class Settings(BaseSettings):
    """AmorphNet Application Settings."""

    # -------------------------------------------------------------------------
    # Cloud Service API Keys & Endpoints
    # -------------------------------------------------------------------------
    LLAMA_CLOUD_API_KEY: str = Field(default="", description="LlamaParse Cloud API Key")
    VOYAGE_API_KEY: str = Field(default="", description="Voyage AI API Key")

    QDRANT_URL: str = Field(default="", description="Qdrant Cloud Cluster URL")
    QDRANT_API_KEY: str = Field(default="", description="Qdrant Cloud API Key")
    QDRANT_COLLECTION: str = Field(default="financial_chunks", description="Qdrant Collection Name")

    NEO4J_URI: str = Field(default="", description="Neo4j AuraDB URI (neo4j+s://...)")
    NEO4J_USERNAME: str = Field(default="neo4j", description="Neo4j AuraDB Username")
    NEO4J_PASSWORD: str = Field(default="", description="Neo4j AuraDB Password")

    GOOGLE_API_KEY: str = Field(default="", description="Google Gemini API Key")

    UPSTASH_REDIS_REST_URL: str = Field(default="", description="Upstash Redis REST URL")
    UPSTASH_REDIS_REST_TOKEN: str = Field(default="", description="Upstash Redis REST Token")
    REDIS_URL: str = Field(default="", description="Optional direct Redis connection URI")

    # -------------------------------------------------------------------------
    # Application & Concurrency Parameters
    # -------------------------------------------------------------------------
    LOG_LEVEL: str = Field(default="INFO", description="Logging Level")
    MAX_CONCURRENT_INGESTIONS: int = Field(default=3, description="Max concurrent background ingestion tasks")
    QUERY_TIMEOUT_SECONDS: int = Field(default=30, description="End-to-end query timeout in seconds")
    CACHE_TTL_SECONDS: int = Field(default=3600, description="Query cache TTL in Redis (1 hour)")

    # -------------------------------------------------------------------------
    # Rate Limiting & Quota Controls
    # -------------------------------------------------------------------------
    LLAMAPARSE_MAX_CONCURRENCY: int = Field(default=3, description="Max concurrent LlamaParse jobs")
    VOYAGE_MAX_RETRIES: int = Field(default=5, description="Max retries for Voyage AI client calls")
    VOYAGE_RPM_LIMIT: int = Field(default=50, description="Voyage AI requests per minute limiter")
    GEMINI_RPM_LIMIT: int = Field(default=14, description="Gemini requests per minute limiter (14 for Free Tier)")
    EVAL_MODE: Literal["smoke", "full"] = Field(default="smoke", description="Evaluation mode (smoke or full)")

    # -------------------------------------------------------------------------
    # Pydantic Settings Configuration
    # -------------------------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    def validate_service_keys(self) -> dict[str, bool]:
        """Check presence of configured credentials for all external cloud services."""
        return {
            "llamaparse": bool(self.LLAMA_CLOUD_API_KEY.strip()),
            "voyage": bool(self.VOYAGE_API_KEY.strip()),
            "qdrant": bool(self.QDRANT_URL.strip() and self.QDRANT_API_KEY.strip()),
            "neo4j": bool(self.NEO4J_URI.strip() and self.NEO4J_PASSWORD.strip()),
            "google_genai": bool(self.GOOGLE_API_KEY.strip()),
            "upstash_redis": bool(
                (self.UPSTASH_REDIS_REST_URL.strip() and self.UPSTASH_REDIS_REST_TOKEN.strip())
                or self.REDIS_URL.strip()
            ),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retrieve cached application settings singleton."""
    return Settings()
