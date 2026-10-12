"""Phase 0.3 — ``amorphnet.config.Settings``: defaults, env parsing, credential validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from amorphnet import config
from amorphnet.config import Settings, get_settings


@pytest.fixture
def settings(clean_env):
    return lambda **kw: Settings(_env_file=None, **kw)


def test_runtime_constants_match_implementation_plan(settings):
    s = settings()
    assert s.GEMINI_RPM_LIMIT == 14
    assert s.MAX_CONCURRENT_INGESTIONS == 3
    assert s.QUERY_TIMEOUT_SECONDS == 30
    assert s.CACHE_TTL_SECONDS == 3600
    assert s.LLAMAPARSE_MAX_CONCURRENCY == 3
    assert s.VOYAGE_MAX_RETRIES == 5
    assert s.VOYAGE_RPM_LIMIT == 50
    assert s.EVAL_MODE == "smoke"
    assert s.QDRANT_COLLECTION == "financial_chunks"
    assert s.NEO4J_USERNAME == "neo4j"
    assert s.LOG_LEVEL == "INFO"


def test_all_credentials_default_to_empty(settings):
    s = settings()
    for name in ("LLAMA_CLOUD_API_KEY", "VOYAGE_API_KEY", "QDRANT_URL", "QDRANT_API_KEY",
                 "NEO4J_URI", "NEO4J_PASSWORD", "GOOGLE_API_KEY", "UPSTASH_REDIS_REST_URL",
                 "UPSTASH_REDIS_REST_TOKEN", "REDIS_URL"):
        assert getattr(s, name) == ""


def test_environment_variables_override_defaults_and_are_typed(clean_env):
    clean_env.setenv("GEMINI_RPM_LIMIT", "1000")
    clean_env.setenv("EVAL_MODE", "full")
    clean_env.setenv("QDRANT_COLLECTION", "custom")
    s = Settings(_env_file=None)
    assert s.GEMINI_RPM_LIMIT == 1000 and isinstance(s.GEMINI_RPM_LIMIT, int)
    assert s.EVAL_MODE == "full"
    assert s.QDRANT_COLLECTION == "custom"


@pytest.mark.parametrize("bad", ["", "medium", "FULL"])
def test_eval_mode_rejects_unknown_values(clean_env, bad):
    clean_env.setenv("EVAL_MODE", bad)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_non_integer_numeric_setting_is_rejected(clean_env):
    clean_env.setenv("QUERY_TIMEOUT_SECONDS", "soon")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_env_names_are_case_sensitive(clean_env):
    clean_env.setenv("gemini_rpm_limit", "1")
    assert Settings(_env_file=None).GEMINI_RPM_LIMIT == 14


def test_unknown_env_vars_are_ignored(clean_env):
    clean_env.setenv("SOMETHING_ELSE", "x")
    Settings(_env_file=None)  # must not raise (extra="ignore")


def test_validate_service_keys_all_missing(settings):
    assert settings().validate_service_keys() == {
        "llamaparse": False, "voyage": False, "qdrant": False,
        "neo4j": False, "google_genai": False, "upstash_redis": False,
    }


def test_validate_service_keys_requires_complete_pairs(settings):
    assert settings(QDRANT_URL="https://x").validate_service_keys()["qdrant"] is False
    assert settings(QDRANT_URL="https://x", QDRANT_API_KEY="k").validate_service_keys()["qdrant"] is True
    assert settings(NEO4J_URI="neo4j+s://x").validate_service_keys()["neo4j"] is False
    assert settings(NEO4J_URI="neo4j+s://x", NEO4J_PASSWORD="p").validate_service_keys()["neo4j"] is True


def test_upstash_accepts_rest_pair_or_redis_url(settings):
    assert settings(UPSTASH_REDIS_REST_URL="u").validate_service_keys()["upstash_redis"] is False
    assert settings(UPSTASH_REDIS_REST_URL="u", UPSTASH_REDIS_REST_TOKEN="t"
                    ).validate_service_keys()["upstash_redis"] is True
    assert settings(REDIS_URL="rediss://x").validate_service_keys()["upstash_redis"] is True


def test_whitespace_only_keys_count_as_missing(settings):
    s = settings(VOYAGE_API_KEY="   ", GOOGLE_API_KEY="\t", LLAMA_CLOUD_API_KEY=" ")
    keys = s.validate_service_keys()
    assert not keys["voyage"] and not keys["google_genai"] and not keys["llamaparse"]


def test_get_settings_is_cached_singleton():
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()


def test_env_example_documents_every_setting():
    """``.env.example`` must list every Settings field so new contributors can configure it."""
    from pathlib import Path

    text = (Path(config.__file__).resolve().parents[1] / ".env.example").read_text()
    for field in Settings.model_fields:
        assert field in text, f"{field} missing from .env.example"
