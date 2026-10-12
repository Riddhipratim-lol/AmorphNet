"""Phase 4.1 — ``amorphnet/ingestion/document_registry.py`` (SQLite-backed registry + dedup gate).

Contract:
  ``compute_sha256(data: bytes) -> str``  (hex digest)
  ``DuplicateDocumentError(existing_document_id)`` with ``.existing_document_id``
  ``DocumentRegistry(db_path=":memory:")`` (sync API):
    ``register(filename, pdf_bytes, company_name, fiscal_year, filing_type, page_count=0)
       -> DocumentRecord`` (raises ``DuplicateDocumentError`` on a repeated SHA-256)
    ``get(document_id) -> DocumentRecord | None``;  ``find_by_hash(sha) -> DocumentRecord | None``
    ``update_status(document_id, status, chunks_indexed=None)`` (invalid status -> ``ValueError``,
       unknown id -> ``KeyError``);  ``list_documents() -> list[DocumentRecord]``
  ``DocumentRecord``: document_id (uuid str), filename, sha256_hash, company_name, fiscal_year,
    filing_type, page_count, ingestion_date (datetime), status, chunks_indexed.
  ``VALID_STATUSES == ("pending","parsing","chunking","enriching","indexed","failed")``
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime

import pytest

from tests.support import require

reg = require("amorphnet.ingestion.document_registry", "DocumentRegistry", "DuplicateDocumentError",
              "compute_sha256", "VALID_STATUSES")

PDF = b"%PDF-1.7 fake bytes A"


@pytest.fixture
def registry():
    return reg.DocumentRegistry(":memory:")


def _register(registry, data=PDF, **kw):
    args = dict(filename="aapl.pdf", company_name="Apple Inc.", fiscal_year=2025, filing_type="10-K")
    args.update(kw)
    return registry.register(args.pop("filename"), data, **args)


def test_sha256_matches_hashlib():
    assert reg.compute_sha256(PDF) == hashlib.sha256(PDF).hexdigest()
    assert reg.compute_sha256(b"") == hashlib.sha256(b"").hexdigest()


def test_valid_statuses():
    assert tuple(reg.VALID_STATUSES) == ("pending", "parsing", "chunking", "enriching", "indexed", "failed")


def test_register_returns_complete_pending_record(registry):
    rec = _register(registry, page_count=120)
    uuid.UUID(rec.document_id)
    assert rec.status == "pending" and rec.chunks_indexed == 0 and rec.page_count == 120
    assert rec.sha256_hash == hashlib.sha256(PDF).hexdigest()
    assert (rec.filename, rec.company_name, rec.fiscal_year, rec.filing_type) == (
        "aapl.pdf", "Apple Inc.", 2025, "10-K")
    assert isinstance(rec.ingestion_date, datetime)


def test_duplicate_bytes_are_rejected_with_existing_id(registry):
    first = _register(registry)
    with pytest.raises(reg.DuplicateDocumentError) as err:
        _register(registry, filename="renamed-copy.pdf", fiscal_year=1999)  # same bytes
    assert err.value.existing_document_id == first.document_id
    assert len(registry.list_documents()) == 1


def test_different_bytes_register_separately(registry):
    a, b = _register(registry), _register(registry, data=PDF + b"B")
    assert a.document_id != b.document_id and len(registry.list_documents()) == 2


def test_get_and_find_by_hash(registry):
    rec = _register(registry)
    assert registry.get(rec.document_id).document_id == rec.document_id
    assert registry.find_by_hash(rec.sha256_hash).document_id == rec.document_id
    assert registry.get("missing") is None and registry.find_by_hash("deadbeef") is None


@pytest.mark.parametrize("status", ["parsing", "chunking", "enriching", "indexed", "failed"])
def test_status_transitions_persist(registry, status):
    rec = _register(registry)
    registry.update_status(rec.document_id, status)
    assert registry.get(rec.document_id).status == status


def test_update_status_with_chunk_count(registry):
    rec = _register(registry)
    registry.update_status(rec.document_id, "indexed", chunks_indexed=212)
    got = registry.get(rec.document_id)
    assert got.status == "indexed" and got.chunks_indexed == 212


def test_update_status_validation(registry):
    rec = _register(registry)
    with pytest.raises(ValueError):
        registry.update_status(rec.document_id, "queued")
    with pytest.raises(KeyError):
        registry.update_status("nope", "indexed")
    assert registry.get(rec.document_id).status == "pending"


def test_registry_persists_to_disk_and_dedups_across_instances(tmp_path):
    path = tmp_path / "registry.db"
    first = reg.DocumentRegistry(str(path))
    rec = _register(first)
    first.update_status(rec.document_id, "indexed", chunks_indexed=3)
    second = reg.DocumentRegistry(str(path))
    assert second.get(rec.document_id).chunks_indexed == 3
    with pytest.raises(reg.DuplicateDocumentError):
        _register(second)


def test_sql_injection_in_metadata_is_inert(registry):
    rec = _register(registry, company_name="x'); DROP TABLE documents;--")
    assert registry.get(rec.document_id).company_name == "x'); DROP TABLE documents;--"
