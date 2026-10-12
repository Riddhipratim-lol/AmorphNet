"""Phase 2.1 — ``amorphnet/ingestion/baseline_ingest.py``.

Contract:
  ``extract_pages(pdf: bytes | str | Path) -> list[PageText]`` (``.page_number`` 1-based, ``.text``;
  blank pages skipped).
  ``chunk_pages(pages, document_id, *, count_tokens=None, chunk_size=512, overlap=50) -> list[dict]``
  with EXACTLY keys ``document_id, page_number, chunk_id, text``; fixed sliding window over
  whitespace-delimited units measured by ``count_tokens``; chunks never span pages.
"""

from __future__ import annotations

import pytest

from tests.support import make_text_pdf, require, word_count

ingest = require("amorphnet.ingestion.baseline_ingest", "extract_pages", "chunk_pages")


def _words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_extract_pages_returns_numbered_text_and_skips_blank_pages():
    pdf = make_text_pdf(["Net sales were high.", "", "Operating income grew."])
    pages = ingest.extract_pages(pdf)
    assert [p.page_number for p in pages] == [1, 3]
    assert "Net sales" in pages[0].text and "Operating income" in pages[1].text


def test_extract_pages_accepts_path(tmp_path):
    f = tmp_path / "a.pdf"
    f.write_bytes(make_text_pdf(["hello world"]))
    assert ingest.extract_pages(f)[0].text.strip() == "hello world"


def test_extract_pages_rejects_garbage():
    with pytest.raises(Exception):
        ingest.extract_pages(b"not a pdf")


def test_real_filing_extracts_text():
    from pathlib import Path

    pdf = Path(__file__).resolve().parents[2] / "data/filings/10-K/AAPL_2025_10K.pdf"
    if not pdf.exists():
        pytest.skip("sample filing missing")
    pages = ingest.extract_pages(pdf)
    assert len(pages) > 50 and any("Apple" in p.text for p in pages)


def test_chunk_pages_fixed_window_and_overlap():
    page = type("P", (), {"page_number": 4, "text": _words(1200)})()
    chunks = ingest.chunk_pages([page], "doc-9", count_tokens=word_count)
    assert len(chunks) == 3
    sizes = [word_count(c["text"]) for c in chunks]
    assert sizes[0] == sizes[1] == 512 and 0 < sizes[2] <= 512
    for a, b in zip(chunks, chunks[1:]):
        assert b["text"].split()[:50] == a["text"].split()[-50:]


def test_chunk_pages_metadata_is_minimal_and_ids_unique():
    pages = [type("P", (), {"page_number": n, "text": _words(700, f"p{n}_")})() for n in (1, 2)]
    chunks = ingest.chunk_pages(pages, "doc-1", count_tokens=word_count)
    assert all(set(c) == {"document_id", "page_number", "chunk_id", "text"} for c in chunks)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    assert {c["document_id"] for c in chunks} == {"doc-1"}
    assert {c["page_number"] for c in chunks} == {1, 2}


def test_chunks_never_cross_page_boundaries():
    pages = [type("P", (), {"page_number": n, "text": _words(100, f"p{n}_")})() for n in (1, 2)]
    for c in ingest.chunk_pages(pages, "d", count_tokens=word_count):
        assert len({w.split("_")[0] for w in c["text"].split()}) == 1


def test_short_page_yields_single_chunk_and_empty_input_yields_none():
    page = type("P", (), {"page_number": 1, "text": "tiny page"})()
    assert len(ingest.chunk_pages([page], "d", count_tokens=word_count)) == 1
    assert ingest.chunk_pages([], "d", count_tokens=word_count) == []


def test_chunk_pages_custom_size():
    page = type("P", (), {"page_number": 1, "text": _words(100)})()
    chunks = ingest.chunk_pages([page], "d", count_tokens=word_count, chunk_size=20, overlap=5)
    assert all(word_count(c["text"]) <= 20 for c in chunks) and len(chunks) >= 6

