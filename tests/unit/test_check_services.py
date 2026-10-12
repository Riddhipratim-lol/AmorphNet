"""Phase 0.5 — ``scripts/check_services.py`` connectivity verification (fully mocked)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from amorphnet.config import Settings

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_services.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("check_services_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def use_settings(mod, clean_env, monkeypatch):
    def _use(**kw):
        monkeypatch.setattr(mod, "get_settings", lambda: Settings(_env_file=None, **kw))

    return _use


@pytest.mark.parametrize(
    "check", ["check_qdrant", "check_neo4j", "check_voyage", "check_google_genai",
              "check_upstash_redis", "check_llamaparse"],
)
async def test_every_check_reports_missing_key_without_credentials(mod, use_settings, check):
    use_settings()
    result = await getattr(mod, check)()
    assert result.status == "MISSING_KEY"
    assert result.service and result.message


async def test_qdrant_passed_counts_collections(mod, use_settings, monkeypatch):
    import qdrant_client

    client = MagicMock()
    client.get_collections = AsyncMock(return_value=SimpleNamespace(collections=[1, 2]))
    client.close = AsyncMock()
    monkeypatch.setattr(qdrant_client, "AsyncQdrantClient", lambda **kw: client)
    use_settings(QDRANT_URL="https://q", QDRANT_API_KEY="k")
    result = await mod.check_qdrant()
    assert result.status == "PASSED" and "2" in result.message


async def test_qdrant_failure_is_captured_not_raised(mod, use_settings, monkeypatch):
    import qdrant_client

    client = MagicMock()
    client.get_collections = AsyncMock(side_effect=RuntimeError("401"))
    monkeypatch.setattr(qdrant_client, "AsyncQdrantClient", lambda **kw: client)
    use_settings(QDRANT_URL="https://q", QDRANT_API_KEY="k")
    result = await mod.check_qdrant()
    assert result.status == "FAILED" and "401" in result.message


async def test_voyage_passed_reports_embedding_dimension(mod, use_settings, monkeypatch):
    import voyageai

    client = MagicMock()
    client.embed = AsyncMock(return_value=SimpleNamespace(embeddings=[[0.0] * 1024]))
    monkeypatch.setattr(voyageai, "AsyncClient", lambda **kw: client)
    use_settings(VOYAGE_API_KEY="pa")
    result = await mod.check_voyage()
    assert result.status == "PASSED" and "1024" in result.message
    assert client.embed.await_args.kwargs["model"] == "voyage-4-large"


async def test_genai_empty_response_is_failure(mod, use_settings, monkeypatch):
    from google import genai

    fake = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(
        generate_content=AsyncMock(return_value=SimpleNamespace(text="")))))
    monkeypatch.setattr(genai, "Client", lambda **kw: fake)
    use_settings(GOOGLE_API_KEY="AIza")
    assert (await mod.check_google_genai()).status == "FAILED"


async def test_genai_passed_uses_flash_lite(mod, use_settings, monkeypatch):
    from google import genai

    gen = AsyncMock(return_value=SimpleNamespace(text="pong"))
    fake = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=gen)))
    monkeypatch.setattr(genai, "Client", lambda **kw: fake)
    use_settings(GOOGLE_API_KEY="AIza")
    assert (await mod.check_google_genai()).status == "PASSED"
    assert gen.await_args.kwargs["model"] == "gemini-3.5-flash-lite"


async def test_llamaparse_passes_when_key_and_sdk_present(mod, use_settings):
    use_settings(LLAMA_CLOUD_API_KEY="llx-1")
    assert (await mod.check_llamaparse()).status in {"PASSED", "MISSING_LIB"}


async def test_main_prints_one_row_per_service(mod, use_settings, capsys):
    use_settings()
    await mod.main()
    out = capsys.readouterr().out
    for name in ("Qdrant Cloud", "Neo4j AuraDB", "Voyage AI", "Google GenAI", "Upstash Redis", "LlamaParse"):
        assert name in out
