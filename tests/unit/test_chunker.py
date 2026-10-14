"""Phase 5.2 / 5.3 / 5.4 — ``amorphnet/ingestion/chunker.py`` heterogeneous element-aware chunker.

Contract (all token counting is injectable via ``count_tokens: Callable[[str], int]``; tests use
whitespace word counts; the default must be the Voyage tokenizer):

  Constants: ``TEXT_TARGET_TOKENS=512``, ``TEXT_OVERLAP_TOKENS=50``, ``TEXT_MAX_TOKENS=768``,
             ``TABLE_SMALL_MAX_TOKENS=300``, ``TABLE_MEDIUM_MAX_TOKENS=800``, ``TABLE_LARGE_MAX_ROWS=10``

  ``split_sections(markdown) -> list[tuple[str, str]]``
      ``(header_text, body)`` pairs split on ``##`` / ``###`` headers (header text without ``#``);
      text before the first header gets header ``""``.
  ``split_sentences(text) -> list[str]``   (never splits inside ``$391.0`` / ``5.2%``)
  ``split_text(text, count_tokens=None, target=512, overlap=50, max_tokens=768) -> list[str]``
      Sentence-boundary packing to ``target``; ~``overlap`` tokens of trailing sentences repeated at
      the start of the next piece; no piece ever exceeds ``max_tokens`` (hard cut for run-ons).
  ``parse_markdown_table(table_md) -> (header_lines: list[str], rows: list[str])``
      ``header_lines`` = header row + ``|---|`` separator; ``rows`` = data rows.
  ``chunk_table(table_md, caption="", count_tokens=None) -> list[str]``
      <=300 tokens: one chunk (caption on top).  301-800: row-group split (a group starts at a
      section-label row = non-indented first cell, every other cell empty; indented rows stay with
      their parent); >800: <=10 data rows per chunk.  EVERY chunk starts with the header+separator
      lines (after an optional caption line) and rows are never lost, duplicated, or reordered.
  ``parse_footnote(text) -> tuple[str, str]``  ``("(1)", "Includes ...")``
  ``chunk_page(page, doc_meta, count_tokens=None, *, section_header="") -> list[Chunk]``
      ``page``: ``.page_number``, ``.markdown_content``, ``.element_type_map``
      (``dict[(start,end)] -> "text"|"table"|"chart_description"|"footnote"``; empty map => the
      chunker auto-detects tables and treats the rest as text).  ``doc_meta``: ``document_id,
      company_name, filing_type, fiscal_year`` (+ optional ``ticker, fiscal_quarter``).
      A table's caption is the nearest non-blank line above it starting with ``Table:``; it is
      stored (prefix stripped) in ``table_caption`` and not emitted as separate text.
      Text chunks: ``markdown_content[char_offset_start:char_offset_end].strip() == text.strip()``.
      Footnote chunks: one per footnote line, ``parent_anchor`` = marker, ``parent_chunk_id`` = id of
      the first text chunk whose text contains the marker ("" if none).
  ``chunk_document(pages, doc_meta, count_tokens=None) -> list[Chunk]``
      Pages in order; the active section header carries across pages; ``chunk_index`` counts chunks
      within a section (restarting at 0 for each new section).
"""

from __future__ import annotations

import math
import re
from types import SimpleNamespace

import pytest

from tests.support import require, word_count

ck = require("amorphnet.ingestion.chunker", "split_sections", "split_sentences", "split_text",
             "parse_markdown_table", "chunk_table", "parse_footnote", "chunk_page", "chunk_document",
             "TEXT_TARGET_TOKENS", "TEXT_OVERLAP_TOKENS", "TEXT_MAX_TOKENS",
             "TABLE_SMALL_MAX_TOKENS", "TABLE_MEDIUM_MAX_TOKENS", "TABLE_LARGE_MAX_ROWS")

DOC = dict(document_id="doc-1", company_name="Apple Inc.", ticker="AAPL", filing_type="10-K",
           fiscal_year=2025)
HEADER = ["| Line item | 2025 | 2024 |", "|---|---|---|"]


# ----------------------------------------------------------------------------- helpers
def sentences(n: int, words: int = 10, tag: str = "s") -> str:
    """``n`` sentences of exactly ``words`` words each (unique words, ends with a period)."""
    out = []
    for i in range(n):
        ws = [f"{tag}{i}w{j}" for j in range(words - 1)] + [f"{tag}{i}end."]
        out.append(" ".join(ws))
    return " ".join(out)


def table(rows: list[str]) -> str:
    return "\n".join(HEADER + rows)


def row(label: str, a="100", b="90") -> str:
    return f"| {label} | {a} | {b} |"


def section_row(label: str) -> str:
    return f"| **{label}** | | |"


def data_rows(chunk_text: str) -> list[str]:
    lines = [ln for ln in chunk_text.splitlines() if ln.strip().startswith("|")]
    return lines[2:]


def make_page(md: str, spans=(), number=1):
    """Build a page whose element map is derived from ``(substring, element_type)`` spans."""
    emap = {}
    for sub, etype in spans:
        s = md.index(sub)
        emap[(s, s + len(sub))] = etype
    return SimpleNamespace(page_number=number, markdown_content=md, element_type_map=emap, raw_images=[])


# ----------------------------------------------------------------------------- constants
def test_constants_match_architecture():
    assert (ck.TEXT_TARGET_TOKENS, ck.TEXT_OVERLAP_TOKENS, ck.TEXT_MAX_TOKENS) == (512, 50, 768)
    assert (ck.TABLE_SMALL_MAX_TOKENS, ck.TABLE_MEDIUM_MAX_TOKENS, ck.TABLE_LARGE_MAX_ROWS) == (300, 800, 10)


# ----------------------------------------------------------------------------- 5.2 sections
def test_split_sections_on_h2_and_h3():
    md = "Intro text.\n\n## Item 7. MD&A\nBody A.\n\n### Net Sales\nBody B.\n"
    assert ck.split_sections(md) == [
        ("", "Intro text."), ("Item 7. MD&A", "Body A."), ("Net Sales", "Body B.")]


def test_split_sections_without_headers_or_empty():
    assert ck.split_sections("Only text.") == [("", "Only text.")]
    assert ck.split_sections("") == []


def test_split_sections_ignores_h1_and_h4_markers():
    md = "## Real\nbody\n#### minor heading\nmore"
    secs = ck.split_sections(md)
    assert [h for h, _ in secs] == ["Real"] and "minor heading" in secs[0][1]


# ----------------------------------------------------------------------------- 5.2 sentences
def test_split_sentences_basic_and_decimals():
    text = "Net sales were $391.0 billion in fiscal 2024. Margin grew 5.2% year over year! Why?"
    parts = ck.split_sentences(text)
    assert parts == ["Net sales were $391.0 billion in fiscal 2024.",
                     "Margin grew 5.2% year over year!", "Why?"]


def test_split_sentences_empty():
    assert ck.split_sentences("   ") == []


# ----------------------------------------------------------------------------- 5.2 split_text
def test_short_text_is_single_piece():
    text = sentences(3)
    assert ck.split_text(text, count_tokens=word_count) == [text]


@pytest.mark.parametrize("blank", ["", "   \n  "])
def test_blank_text_yields_nothing(blank):
    assert ck.split_text(blank, count_tokens=word_count) == []


def test_long_text_packs_to_target_and_never_cuts_sentences():
    pieces = ck.split_text(sentences(300), count_tokens=word_count)  # 3000 words
    assert len(pieces) >= 6
    sizes = [word_count(p) for p in pieces]
    assert all(s <= 512 for s in sizes)
    assert all(s >= 400 for s in sizes[:-1])
    assert all(p.strip().endswith(".") for p in pieces)


def test_consecutive_pieces_overlap_by_about_50_tokens():
    pieces = ck.split_text(sentences(300), count_tokens=word_count)
    for a, b in zip(pieces, pieces[1:]):
        aw, bw = a.split(), b.split()
        n = max((k for k in range(1, min(len(aw), len(bw)) + 1) if aw[-k:] == bw[:k]), default=0)
        assert 40 <= n <= 60, f"overlap was {n} tokens"


def test_all_words_are_preserved_across_pieces():
    text = sentences(120)
    pieces = ck.split_text(text, count_tokens=word_count)
    assert set(text.split()) <= {w for p in pieces for w in p.split()}


def test_run_on_text_is_hard_capped_at_768():
    run_on = " ".join(f"w{i}" for i in range(2000))  # no sentence boundary at all
    pieces = ck.split_text(run_on, count_tokens=word_count)
    assert len(pieces) >= 3 and all(word_count(p) <= 768 for p in pieces)
    assert set(run_on.split()) <= {w for p in pieces for w in p.split()}


def test_oversized_single_sentence_among_normal_ones_is_capped():
    text = sentences(5) + " " + " ".join(f"x{i}" for i in range(1000)) + ". " + sentences(5, tag="t")
    assert all(word_count(p) <= 768 for p in ck.split_text(text, count_tokens=word_count))


def test_custom_target_and_overlap():
    pieces = ck.split_text(sentences(100), count_tokens=word_count, target=100, overlap=20, max_tokens=150)
    assert len(pieces) >= 10 and all(word_count(p) <= 100 for p in pieces)


# ----------------------------------------------------------------------------- 5.3 tables
def test_parse_markdown_table():
    header, rows = ck.parse_markdown_table(table([row("Revenue"), row("Costs", "1", "2")]))
    assert header == HEADER and rows == [row("Revenue"), row("Costs", "1", "2")]


def test_small_table_is_one_chunk_with_caption():
    t = table([row(f"Item {i}") for i in range(5)])
    assert word_count(t) <= 300
    chunks = ck.chunk_table(t, caption="Net sales by category", count_tokens=word_count)
    assert len(chunks) == 1
    assert "Net sales by category" in chunks[0] and all(r in chunks[0] for r in data_rows(t))


def test_large_table_splits_into_ten_row_chunks_with_headers_prepended():
    rows = [row(f"Line {i}", str(i), str(i * 2)) for i in range(120)]
    t = table(rows)
    assert word_count(t) > 800
    chunks = ck.chunk_table(t, caption="Big table", count_tokens=word_count)
    assert len(chunks) == math.ceil(120 / 10)
    for c in chunks:
        assert HEADER[0] in c and HEADER[1] in c
        assert 1 <= len(data_rows(c)) <= 10
    assert [r for c in chunks for r in data_rows(c)] == rows  # nothing lost/duplicated/reordered


def _medium_table():
    rows = []
    for cat in ("Revenue", "Expenses", "Assets"):
        rows.append(section_row(cat))
        rows += [row(f"{cat} item {i}") for i in range(12)]
    return table(rows), rows


def test_medium_table_splits_by_semantic_row_group():
    t, rows = _medium_table()
    assert 300 < word_count(t) <= 800
    chunks = ck.chunk_table(t, count_tokens=word_count)
    assert len(chunks) == 3
    for c, cat in zip(chunks, ("Revenue", "Expenses", "Assets")):
        assert HEADER[0] in c and HEADER[1] in c
        body = data_rows(c)
        assert body[0] == section_row(cat)
        assert sum("**" in r for r in body) == 1  # never two categories in one chunk
    assert [r for c in chunks for r in data_rows(c)] == rows


def test_medium_table_keeps_indented_children_with_parent():
    rows = []
    for g in range(6):
        rows.append(section_row(f"Group {g}"))
        rows.append(row(f"Parent {g}"))
        rows += [row(f"   Child {g}.{i}") for i in range(8)]
    chunks = ck.chunk_table(table(rows), count_tokens=word_count)
    assert len(chunks) > 1
    for c in chunks:
        assert not data_rows(c)[0].startswith("|    ")  # no orphaned indented row at chunk start
    assert [r for c in chunks for r in data_rows(c)] == rows


def test_medium_table_without_section_rows_still_splits_losslessly():
    rows = [row(f"Flat {i}") for i in range(40)]
    t = table(rows)
    assert 300 < word_count(t) <= 800
    chunks = ck.chunk_table(t, count_tokens=word_count)
    assert len(chunks) >= 2 and [r for c in chunks for r in data_rows(c)] == rows
    assert all(HEADER[1] in c for c in chunks)


@pytest.mark.parametrize("fake_tokens,expected_kind", [
    (300, "single"), (301, "groups"), (800, "groups"), (801, "rows")])
def test_tier_boundaries(fake_tokens, expected_kind):
    rows = []
    for cat in ("A", "B"):
        rows.append(section_row(cat))
        rows += [row(f"{cat}{i}") for i in range(15)]
    chunks = ck.chunk_table(table(rows), count_tokens=lambda _t: fake_tokens)
    if expected_kind == "single":
        assert len(chunks) == 1
    elif expected_kind == "groups":
        assert len(chunks) == 2  # one per category (16 rows each: group split, not 10-row split)
    else:
        assert len(chunks) >= 4 and all(len(data_rows(c)) <= 10 for c in chunks)


# ----------------------------------------------------------------------------- 5.4 footnotes
@pytest.mark.parametrize("raw,marker,body", [
    ("(1) Includes the impact of foreign currency.", "(1)", "Includes the impact of foreign currency."),
    ("a) Excludes restructuring.", "a)", "Excludes restructuring."),
    ("(12)   Unaudited.", "(12)", "Unaudited."),
])
def test_parse_footnote(raw, marker, body):
    assert ck.parse_footnote(raw) == (marker, body)


# ----------------------------------------------------------------------------- chunk_page
PAGE_MD = (
    "## Item 7. Management's Discussion\n"
    "Net sales increased 6% during fiscal 2025.(1) Services grew fastest across regions.\n\n"
    "Table: Net sales by category\n"
    + table([row("iPhone", "209,586", "201,183"), row("Services", "109,158", "96,169")])
    + "\n\nChart description: Bar chart of net sales by segment for FY2023-2025; Services grows fastest.\n\n"
    "(1) Includes the impact of foreign currency.\n"
    "(2) Excludes restructuring charges."
)
TABLE_BLOCK = table([row("iPhone", "209,586", "201,183"), row("Services", "109,158", "96,169")])
SPANS = (
    ("Net sales increased 6% during fiscal 2025.(1) Services grew fastest across regions.", "text"),
    (TABLE_BLOCK, "table"),
    ("Chart description: Bar chart of net sales by segment for FY2023-2025; Services grows fastest.",
     "chart_description"),
    ("(1) Includes the impact of foreign currency.\n(2) Excludes restructuring charges.", "footnote"),
)


@pytest.fixture
def page_chunks():
    return ck.chunk_page(make_page(PAGE_MD, SPANS, number=42), DOC, count_tokens=word_count)


def by_type(chunks, t):
    return [c for c in chunks if c.element_type == t]


def test_page_yields_all_four_element_types(page_chunks):
    assert {c.element_type for c in page_chunks} == {"text", "table", "chart", "footnote"}
    assert len(by_type(page_chunks, "footnote")) == 2  # one chunk per footnote line
    assert len(by_type(page_chunks, "chart")) == 1


def test_metadata_is_stamped_on_every_chunk(page_chunks):
    for c in page_chunks:
        assert (c.document_id, c.company_name, c.ticker, c.filing_type, c.fiscal_year) == (
            "doc-1", "Apple Inc.", "AAPL", "10-K", 2025)
        assert c.fiscal_quarter == "Annual" and c.page_number == 42
        assert c.section_header == "Item 7. Management's Discussion"
        assert c.token_count == word_count(c.text) > 0
    assert len({c.chunk_id for c in page_chunks}) == len(page_chunks)


def test_text_chunk_offsets_slice_back_to_text(page_chunks):
    for c in by_type(page_chunks, "text"):
        assert PAGE_MD[c.char_offset_start:c.char_offset_end].strip() == c.text.strip()
    for c in page_chunks:
        assert 0 <= c.char_offset_start < c.char_offset_end <= len(PAGE_MD)


def test_table_chunk_has_caption_metadata_and_caption_is_not_duplicated_as_text(page_chunks):
    (t,) = by_type(page_chunks, "table")
    assert t.table_caption == "Net sales by category"
    assert all(r in t.text for r in data_rows(TABLE_BLOCK))
    assert not any("Table: Net sales by category" in c.text for c in by_type(page_chunks, "text"))
    assert all(c.table_caption == "" for c in page_chunks if c.element_type != "table")


def test_chart_chunk_is_single_and_intact(page_chunks):
    (c,) = by_type(page_chunks, "chart")
    assert c.text.startswith("Chart description:") and "Services grows fastest" in c.text


def test_footnote_chunks_link_to_parent_text_chunk(page_chunks):
    foots = {c.parent_anchor: c for c in by_type(page_chunks, "footnote")}
    assert set(foots) == {"(1)", "(2)"}
    (text,) = [c for c in by_type(page_chunks, "text")]
    assert foots["(1)"].parent_chunk_id == text.chunk_id  # marker "(1)" occurs in that text chunk
    assert foots["(2)"].parent_chunk_id == ""  # marker never referenced in narrative
    assert foots["(1)"].text.startswith("(1)") and "foreign currency" in foots["(1)"].text


def test_non_footnote_chunks_have_no_parent_anchor(page_chunks):
    assert all(c.parent_anchor == "" for c in page_chunks if c.element_type != "footnote")


def test_large_table_chunks_all_carry_the_same_caption():
    rows = [row(f"Line {i}") for i in range(60)]
    md = "Table: Giant table\n" + table(rows)
    page = make_page(md, [(table(rows), "table")])
    chunks = ck.chunk_page(page, DOC, count_tokens=word_count)
    assert len(chunks) == 6 and {c.table_caption for c in chunks} == {"Giant table"}
    assert all(c.element_type == "table" for c in chunks)


def test_table_without_caption_has_empty_caption():
    chunks = ck.chunk_page(make_page(table([row("A")]), [(table([row("A")]), "table")]), DOC,
                           count_tokens=word_count)
    assert chunks[0].table_caption == ""


def test_empty_element_map_auto_detects_tables_and_text():
    md = "Prose before the table.\n\n" + table([row("iPhone"), row("Services")]) + "\n\nProse after."
    chunks = ck.chunk_page(make_page(md), DOC, count_tokens=word_count)
    assert sorted(c.element_type for c in chunks) == ["table", "text", "text"]


def test_long_narrative_page_respects_token_limits():
    md = "## Risk Factors\n" + sentences(300)
    chunks = ck.chunk_page(make_page(md, [(sentences(300), "text")]), DOC, count_tokens=word_count)
    assert len(chunks) >= 6 and all(c.token_count <= 768 for c in chunks)
    assert all(c.section_header == "Risk Factors" for c in chunks)


def test_blank_page_yields_no_chunks():
    assert ck.chunk_page(make_page("   \n"), DOC, count_tokens=word_count) == []


def test_missing_optional_doc_meta_uses_defaults():
    meta = {k: v for k, v in DOC.items() if k != "ticker"}
    (c,) = ck.chunk_page(make_page("Hello world."), meta, count_tokens=word_count)
    assert c.ticker == "" and c.fiscal_quarter == "Annual"


def test_doc_meta_fiscal_quarter_override():
    (c,) = ck.chunk_page(make_page("Hello world."), {**DOC, "fiscal_quarter": "Q3"}, count_tokens=word_count)
    assert c.fiscal_quarter == "Q3"


# ----------------------------------------------------------------------------- chunk_document
def _doc_pages():
    p1 = "## Business\n" + sentences(5, tag="a") + "\n\n## Risk Factors\n" + sentences(5, tag="b")
    p2 = sentences(5, tag="c") + "\n\n## Legal Proceedings\n" + sentences(5, tag="d")
    return [make_page(p1, number=1), make_page(p2, number=2)]


def test_section_header_carries_across_pages():
    chunks = ck.chunk_document(_doc_pages(), DOC, count_tokens=word_count)
    page2 = [c for c in chunks if c.page_number == 2]
    assert page2[0].section_header == "Risk Factors"  # inherited: page 2 starts mid-section
    assert page2[-1].section_header == "Legal Proceedings"


def test_chunk_index_counts_within_section_and_restarts():
    chunks = ck.chunk_document(_doc_pages(), DOC, count_tokens=word_count)
    by_section: dict[str, list[int]] = {}
    for c in chunks:
        by_section.setdefault(c.section_header, []).append(c.chunk_index)
    for idx in by_section.values():
        assert idx == list(range(len(idx)))
    assert set(by_section) == {"Business", "Risk Factors", "Legal Proceedings"}


def test_no_overlap_leaks_across_section_boundaries():
    chunks = ck.chunk_document(_doc_pages(), DOC, count_tokens=word_count)
    for c in chunks:
        tags = {re.match(r"[a-z]", w).group(0) for w in c.text.split() if re.match(r"[a-z]\d", w)}
        assert len(tags) == 1, c.text[:60]


def test_document_order_and_no_content_loss():
    pages = _doc_pages()
    chunks = ck.chunk_document(pages, DOC, count_tokens=word_count)
    assert [c.page_number for c in chunks] == sorted(c.page_number for c in chunks)
    source_words = {w for pg in pages for w in pg.markdown_content.split() if re.match(r"[a-z]\d", w)}
    assert source_words <= {w for c in chunks for w in c.text.split()}


def test_document_skips_blank_pages_and_empty_document():
    pages = [make_page("Real text here.", number=1), make_page("  ", number=2)]
    assert [c.page_number for c in ck.chunk_document(pages, DOC, count_tokens=word_count)] == [1]
    assert ck.chunk_document([], DOC, count_tokens=word_count) == []


def test_every_chunk_in_a_mixed_document_stays_under_hard_cap():
    pages = [make_page(PAGE_MD, SPANS, number=1), make_page("## Notes\n" + sentences(400), number=2)]
    assert all(c.token_count <= 768 for c in ck.chunk_document(pages, DOC, count_tokens=word_count))
