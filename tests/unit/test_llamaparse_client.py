"""Phase 4.3 — ``amorphnet/ingestion/llamaparse_client.py``.

Contract:
  ``PARSE_CONFIG`` (dict, exactly the architecture 5.2 values) and ``FINANCIAL_PARSING_INSTRUCTIONS``.
  ``LlamaParseResult(page_number, markdown_content, element_type_map, raw_images)`` dataclass;
      ``element_type_map: dict[tuple[int, int], str]`` mapping ``(start, end)`` char offsets of
      ``markdown_content`` to ``"text" | "table" | "chart_description" | "footnote"``.
  ``classify_elements(markdown) -> dict[tuple[int,int], str]`` — non-overlapping, trimmed ranges:
      markdown table blocks => ``table``; paragraphs starting ``Chart description:`` =>
      ``chart_description``; lines starting with a ``(1)`` / ``a)`` marker => ``footnote``;
      everything else => ``text``.
  ``split_pages(raw_markdown, start_page=1) -> list[LlamaParseResult]`` splitting on
      ``---PAGE n---`` separators; ``page_number = n + start_page - 1``.
  ``LlamaParseClient(settings=None, *, backend=None, max_concurrency=None)``
      ``backend``: ``async parse(pdf_bytes, filename, config) -> str`` (raw markdown with separators)
      ``async parse_pdf(pdf_bytes, *, filename="document.pdf", start_page=1) -> list[LlamaParseResult]``
         — retried with exponential back-off (2s..30s) on HTTP 429 only, max 5 attempts.
      ``async parse_batches(batches, *, filename="document.pdf") -> list[LlamaParseResult]``
         — ``asyncio.Semaphore(max_concurrency=3)``; results sorted by page number.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from amorphnet.config import Settings
from tests.support import require

lp = require("amorphnet.ingestion.llamaparse_client", "PARSE_CONFIG", "FINANCIAL_PARSING_INSTRUCTIONS",
             "LlamaParseResult", "LlamaParseClient", "split_pages", "classify_elements")

SAMPLE = """---PAGE 1---
## Consolidated Statements of Operations

Net sales increased 6% during fiscal 2025.(1)

Table: Net sales by category
| Category | 2025 | 2024 |
|---|---|---|
| iPhone | 209,586 | 201,183 |
| Services | 109,158 | 96,169 |

Chart description: Bar chart of net sales by segment for FY2023-2025; Services grows fastest.

(1) Includes the impact of foreign currency.
---PAGE 2---
Second page narrative about risk factors.
"""


def _http_error(status: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "https://api.cloud.llamaindex.ai/parse")
    return httpx.HTTPStatusError("err", request=req, response=httpx.Response(status, request=req))


class Backend:
    def __init__(self, behaviour=None, delay=0.0):
        self.calls, self.behaviour, self.delay = [], behaviour, delay
        self.in_flight = self.max_in_flight = 0

    async def parse(self, pdf_bytes, filename, config):
        self.calls.append({"pdf": pdf_bytes, "filename": filename, "config": config})
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            b = self.behaviour
            if callable(b):
                return b(len(self.calls), pdf_bytes)
            return b if b is not None else "---PAGE 1---\nhello"
        finally:
            self.in_flight -= 1


def _client(backend, **kw):
    return lp.LlamaParseClient(Settings(_env_file=None, LLAMA_CLOUD_API_KEY="llx-x"), backend=backend, **kw)


# ------------------------------------------------------------------ configuration
def test_parse_config_matches_architecture():
    cfg = lp.PARSE_CONFIG
    assert cfg["parse_mode"] == "parse" and cfg["output_format"] == "markdown"
    assert cfg["language"] == "en" and cfg["premium_mode"] is True
    assert cfg["skip_diagonal_text"] is True and cfg["take_screenshot"] is False
    assert cfg["invalidate_cache"] is False  # keep the 48h LlamaParse cache
    assert cfg["page_separator"] == "---PAGE {page_num}---"


def test_financial_instructions_cover_every_rule():
    text = lp.FINANCIAL_PARSING_INSTRUCTIONS.lower()
    for needle in ("table", "markdown", "caption", "chart", "footnote", "left column",
                   "currency", "in millions", "line item"):
        assert needle in text, needle


# ------------------------------------------------------------------ page splitting
def test_split_pages_numbers_and_content():
    pages = lp.split_pages(SAMPLE)
    assert [p.page_number for p in pages] == [1, 2]
    assert "Consolidated Statements" in pages[0].markdown_content
    assert "---PAGE" not in pages[0].markdown_content
    assert pages[1].markdown_content.strip().startswith("Second page")
    assert pages[0].raw_images == []


def test_split_pages_applies_start_page_offset():
    assert [p.page_number for p in lp.split_pages(SAMPLE, start_page=201)] == [201, 202]


def test_split_pages_empty_input():
    assert lp.split_pages("") == []


# ------------------------------------------------------------------ element classification
def test_classify_elements_identifies_all_four_types():
    page = lp.split_pages(SAMPLE)[0]
    md, emap = page.markdown_content, page.element_type_map
    by_type: dict[str, list[str]] = {}
    for (s, e), t in emap.items():
        by_type.setdefault(t, []).append(md[s:e])
    assert set(by_type) == {"text", "table", "chart_description", "footnote"}
    assert len(by_type["table"]) == 1
    tbl = by_type["table"][0]
    assert tbl.startswith("| Category") and tbl.endswith("| 96,169 |") and tbl.count("\n") == 3
    assert by_type["chart_description"][0].startswith("Chart description:")
    assert by_type["footnote"] == ["(1) Includes the impact of foreign currency."]
    assert any("Net sales increased 6%" in t for t in by_type["text"])


def test_classified_ranges_are_trimmed_and_non_overlapping():
    page = lp.split_pages(SAMPLE)[0]
    ranges = sorted(page.element_type_map)
    for s, e in ranges:
        assert 0 <= s < e <= len(page.markdown_content)
        seg = page.markdown_content[s:e]
        assert seg == seg.strip()
    assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))


def test_table_caption_line_is_not_part_of_table_range():
    page = lp.split_pages(SAMPLE)[0]
    for (s, e), t in page.element_type_map.items():
        if t == "table":
            assert "Table: Net sales by category" not in page.markdown_content[s:e]


def test_letter_footnote_markers_and_plain_page():
    md = "Body text.\n\na) Excludes restructuring.\nb) Unaudited."
    emap = lp.classify_elements(md)
    foot = [md[s:e] for (s, e), t in emap.items() if t == "footnote"]
    assert foot == ["a) Excludes restructuring.", "b) Unaudited."]
    assert set(lp.classify_elements("Just prose.").values()) == {"text"}


def test_numbered_list_prose_is_not_a_footnote():
    emap = lp.classify_elements("We rely on:\n1. suppliers\n2. partners")
    assert "footnote" not in set(emap.values())


# ------------------------------------------------------------------ client
async def test_parse_pdf_sends_domain_config_and_returns_pages():
    be = Backend(SAMPLE)
    pages = await _client(be).parse_pdf(b"PDF", filename="aapl.pdf")
    call = be.calls[0]
    assert call["pdf"] == b"PDF" and call["filename"] == "aapl.pdf"
    for k, v in lp.PARSE_CONFIG.items():
        assert call["config"][k] == v
    assert "footnote" in str(call["config"].get("parsing_instruction", "")).lower()
    assert [p.page_number for p in pages] == [1, 2]


async def test_429_is_retried_with_exponential_backoff(no_sleep):
    be = Backend(lambda n, _: (_ for _ in ()).throw(_http_error(429)) if n < 3 else SAMPLE)
    pages = await _client(be).parse_pdf(b"PDF")
    assert len(be.calls) == 3 and len(pages) == 2
    assert no_sleep[0] == 2 and all(2 <= d <= 30 for d in no_sleep)
    assert no_sleep == sorted(no_sleep)


async def test_429_gives_up_after_five_attempts(no_sleep):
    be = Backend(lambda n, _: (_ for _ in ()).throw(_http_error(429)))
    with pytest.raises(httpx.HTTPStatusError):
        await _client(be).parse_pdf(b"PDF")
    assert len(be.calls) == 5


@pytest.mark.parametrize("status", [400, 401, 500])
async def test_non_429_errors_fail_fast(no_sleep, status):
    be = Backend(lambda n, _: (_ for _ in ()).throw(_http_error(status)))
    with pytest.raises(httpx.HTTPStatusError):
        await _client(be).parse_pdf(b"PDF")
    assert len(be.calls) == 1 and no_sleep == []


async def test_parse_batches_caps_concurrency_at_three_and_orders_pages():
    def behaviour(n, pdf):
        return "---PAGE 1---\nbatch " + pdf.decode()

    be = Backend(behaviour, delay=0.02)
    batches = [SimpleNamespace(data=str(i).encode(), start_page=1 + i * 200, end_page=200 * (i + 1),
                               page_count=200) for i in range(8)]
    pages = await _client(be).parse_batches(batches)
    assert be.max_in_flight == 3
    assert [p.page_number for p in pages] == [1 + i * 200 for i in range(8)]
    assert [p.markdown_content.strip() for p in pages] == [f"batch {i}" for i in range(8)]


async def test_max_concurrency_defaults_from_settings_and_is_overridable():
    be = Backend(delay=0.02)
    batches = [SimpleNamespace(data=b"x", start_page=i + 1, end_page=i + 1, page_count=1) for i in range(6)]
    await _client(be, max_concurrency=1).parse_batches(batches)
    assert be.max_in_flight == 1


def test_client_without_credentials_or_backend_fails_fast(clean_env):
    with pytest.raises(ValueError):
        lp.LlamaParseClient(Settings(_env_file=None))
