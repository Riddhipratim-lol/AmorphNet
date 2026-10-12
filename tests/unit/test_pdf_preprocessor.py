"""Phase 4.2 — ``amorphnet/ingestion/pdf_preprocessor.py``.

Contract:
  ``MAX_PAGES == 500``, ``BATCH_PAGES == 200``
  ``EncryptedPDFError`` / ``InvalidPDFError`` (both ``Exception`` subclasses)
  ``inspect_pdf(pdf_bytes) -> PdfInfo`` (``page_count, scan_mode, title, author, creation_date``)
  ``split_pdf(pdf_bytes, max_pages=500, batch_size=200) -> list[PdfBatch]``
      ``PdfBatch``: ``data`` (bytes of a valid PDF), ``start_page`` (1-based, in original doc),
      ``end_page`` (inclusive), ``page_count``.  <= ``max_pages`` pages => exactly one batch
      (whole document); otherwise sequential batches of <= ``batch_size`` pages covering every page.
  ``preprocess_pdf(pdf_bytes) -> PreprocessResult`` (``.info: PdfInfo``, ``.batches: list[PdfBatch]``)
"""

from __future__ import annotations

import pymupdf
import pytest

from tests.support import make_encrypted_pdf, make_scanned_pdf, make_text_pdf, require

pre = require("amorphnet.ingestion.pdf_preprocessor", "inspect_pdf", "split_pdf", "preprocess_pdf",
              "EncryptedPDFError", "InvalidPDFError", "MAX_PAGES", "BATCH_PAGES")


def _pages(data: bytes) -> int:
    with pymupdf.open(stream=data, filetype="pdf") as d:
        return d.page_count


def test_limits_match_llamaparse_ceilings():
    assert pre.MAX_PAGES == 500 and pre.BATCH_PAGES == 200


def test_inspect_text_pdf_basic_metadata():
    info = pre.inspect_pdf(make_text_pdf(4, title="Apple 10-K", author="Apple Inc."))
    assert info.page_count == 4 and info.scan_mode is False
    assert info.title == "Apple 10-K" and info.author == "Apple Inc."


def test_inspect_handles_missing_metadata():
    info = pre.inspect_pdf(make_text_pdf(1))
    assert info.title in ("", None) and info.author in ("", None)


def test_scanned_pdf_sets_scan_mode():
    info = pre.inspect_pdf(make_scanned_pdf(3))
    assert info.scan_mode is True and info.page_count == 3


def test_mixed_pdf_is_not_scan_mode():
    import pymupdf as fz

    doc = fz.open()
    pix = fz.Pixmap(fz.csRGB, fz.IRect(0, 0, 10, 10), False)
    pix.clear_with(1)
    for i in range(4):
        p = doc.new_page()
        if i % 2 == 0:
            p.insert_text((72, 72), "real text here")
        else:
            p.insert_image(fz.Rect(0, 0, 50, 50), pixmap=pix)
    assert pre.inspect_pdf(doc.tobytes()).scan_mode is False


def test_encrypted_pdf_raises_explicit_error():
    with pytest.raises(pre.EncryptedPDFError):
        pre.inspect_pdf(make_encrypted_pdf())
    with pytest.raises(pre.EncryptedPDFError):
        pre.preprocess_pdf(make_encrypted_pdf())


@pytest.mark.parametrize("junk", [b"", b"definitely not a pdf"])
def test_invalid_pdf_raises(junk):
    with pytest.raises(pre.InvalidPDFError):
        pre.inspect_pdf(junk)


def test_small_document_is_a_single_whole_batch():
    data = make_text_pdf(30)
    batches = pre.split_pdf(data)
    assert len(batches) == 1
    b = batches[0]
    assert (b.start_page, b.end_page, b.page_count) == (1, 30, 30) and _pages(b.data) == 30


def test_exactly_500_pages_is_not_split():
    assert len(pre.split_pdf(make_text_pdf(500))) == 1


def test_over_500_pages_splits_into_sequential_200_page_batches():
    batches = pre.split_pdf(make_text_pdf(501))
    assert [b.page_count for b in batches] == [200, 200, 101]
    assert [(b.start_page, b.end_page) for b in batches] == [(1, 200), (201, 400), (401, 501)]
    assert all(_pages(b.data) == b.page_count for b in batches)


def test_split_preserves_page_content_order():
    data = make_text_pdf([f"UNIQUE-{i}" for i in range(1, 8)])
    batches = pre.split_pdf(data, max_pages=5, batch_size=3)
    texts = []
    for b in batches:
        with pymupdf.open(stream=b.data, filetype="pdf") as d:
            texts += [p.get_text().strip() for p in d]
    assert texts == [f"UNIQUE-{i}" for i in range(1, 8)]


def test_preprocess_pdf_combines_info_and_batches():
    res = pre.preprocess_pdf(make_text_pdf(10))
    assert res.info.page_count == 10 and len(res.batches) == 1
