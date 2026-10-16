"""Phase 6 — ``amorphnet/ingestion/metadata_enricher.py`` (batched enrichment + breadcrumbs).

Contract:
  Schemas (pydantic):
    ``ChunkMetadata(chunk_id, financial_metrics: list[str], fiscal_periods: list[str],
                    currency_and_scale: str | None, is_comparative: bool)``
    ``BatchEnrichmentResponse(enriched_chunks: list[ChunkMetadata])``
  ``MIN_BATCH_SIZE == 6``, ``MAX_BATCH_SIZE == 8``, ``DEFAULT_BATCH_SIZE == 8``
  ``batch_chunks(chunks, batch_size=8) -> list[list[Chunk]]`` — consecutive, order-preserving,
      every batch <= ``batch_size``; ``batch_size`` outside 6..8 => ``ValueError``.
  ``ENRICHMENT_SYSTEM_PROMPT`` — instructs entity extraction and scale preservation.
  ``async enrich_chunks(chunks, *, client=None, limiter=None, strict=False) -> list[Chunk]``
      One Gemini call per batch: model ``gemini-3.5-flash-lite``; ``contents`` holds every
      ``chunk_id`` and text of the batch; config: ``response_schema=BatchEnrichmentResponse``,
      ``response_mime_type="application/json"``, ``system_instruction=ENRICHMENT_SYSTEM_PROMPT``.
      ``limiter`` (default ``clients.gemini_limiter``) is acquired once per API attempt.
      HTTP 429 (``APIError``) is retried with exponential back-off, max 5 attempts.
      Returned list has the same length/order as the input; results are merged by ``chunk_id``
      (unknown ids ignored, chunks missing from the response left untouched).
      A batch that still fails (retries exhausted / invalid JSON) leaves its chunks un-enriched and
      logs an error, unless ``strict=True`` which re-raises.
  Section hierarchy:
    ``extract_headings(pages: list[tuple[int, str]]) -> list[Heading]``
        ``Heading(level, text, page_number, breadcrumb)``; markdown ``#``..``######`` headings in
        document order; ``breadcrumb`` joins ancestor headings with `` > ``.
    ``build_section_tree(headings, document_id) -> list[SectionNode]``
        ``SectionNode(section_id, document_id, header_text, breadcrumb, page_number, level,
        parent_id, element_type="text")`` in document order; ``parent_id`` is ``None`` for roots.
    ``section_for_chunk(chunk, sections) -> SectionNode | None`` — the section whose
        ``header_text == chunk.section_header`` with the greatest ``page_number <= chunk.page_number``.
    ``assign_breadcrumbs(chunks, sections) -> list[Chunk]`` — sets ``chunk.breadcrumb``; chunks whose
        header matches no section fall back to ``section_header``; empty header => ``""``.
"""

from __future__ import annotations

import json
import re

import pytest

from tests.support import FakeGenai, genai_error, require

en = require("amorphnet.ingestion.metadata_enricher", "ChunkMetadata", "BatchEnrichmentResponse",
             "batch_chunks", "enrich_chunks", "ENRICHMENT_SYSTEM_PROMPT", "extract_headings",
             "build_section_tree", "section_for_chunk", "assign_breadcrumbs", "SectionNode",
             "MIN_BATCH_SIZE", "MAX_BATCH_SIZE", "DEFAULT_BATCH_SIZE")


# ----------------------------------------------------------------------------- helpers
def _meta(chunk_id, **kw):
    base = dict(chunk_id=chunk_id, financial_metrics=["revenue"], fiscal_periods=["FY2024"],
                currency_and_scale="USD in millions", is_comparative=True)
    base.update(kw)
    return en.ChunkMetadata(**base)


def responder_for(overrides=None, skip=()):
    """Gemini responder that enriches every chunk id mentioned in ``contents``."""
    overrides = overrides or {}

    def respond(call):
        ids = re.findall(r"chunk-\d+|extra-\w+", str(call["contents"]))
        seen = list(dict.fromkeys(ids))
        items = [overrides.get(i, _meta(i)) for i in seen if i not in skip]
        return en.BatchEnrichmentResponse(enriched_chunks=items).model_dump_json()

    return respond


@pytest.fixture
def chunks(make_chunk):
    return [make_chunk(chunk_id=f"chunk-{i}", text=f"Net sales body number {i}.") for i in range(20)]


# ----------------------------------------------------------------------------- schemas
def test_batch_size_constants():
    assert (en.MIN_BATCH_SIZE, en.MAX_BATCH_SIZE, en.DEFAULT_BATCH_SIZE) == (6, 8, 8)


def test_schemas_round_trip_json():
    resp = en.BatchEnrichmentResponse(enriched_chunks=[_meta("a"), _meta("b", currency_and_scale=None)])
    back = en.BatchEnrichmentResponse.model_validate_json(resp.model_dump_json())
    assert back == resp and back.enriched_chunks[1].currency_and_scale is None


def test_chunk_metadata_requires_chunk_id_and_types():
    with pytest.raises(Exception):
        en.ChunkMetadata(financial_metrics=[], fiscal_periods=[], currency_and_scale=None,
                         is_comparative=False)
    with pytest.raises(Exception):
        _meta("a", is_comparative="maybe")


def test_response_schema_is_usable_as_gemini_response_schema():
    schema = en.BatchEnrichmentResponse.model_json_schema()
    assert "enriched_chunks" in schema["properties"]


def test_system_prompt_mentions_entities_and_scale():
    p = en.ENRICHMENT_SYSTEM_PROMPT.lower()
    assert "metric" in p and "period" in p and "scale" in p and "usd in millions" in p


# ----------------------------------------------------------------------------- batching
def test_batch_chunks_default_is_eight_consecutive(chunks):
    batches = en.batch_chunks(chunks)
    assert [len(b) for b in batches] == [8, 8, 4]
    assert [c.chunk_id for b in batches for c in b] == [c.chunk_id for c in chunks]


@pytest.mark.parametrize("size", [6, 7, 8])
def test_batch_chunks_allowed_sizes(chunks, size):
    batches = en.batch_chunks(chunks, size)
    assert all(len(b) <= size for b in batches) and len(batches) == -(-20 // size)


@pytest.mark.parametrize("size", [0, 1, 5, 9, 50])
def test_batch_chunks_rejects_sizes_outside_6_to_8(chunks, size):
    with pytest.raises(ValueError):
        en.batch_chunks(chunks, size)


def test_batch_chunks_empty():
    assert en.batch_chunks([]) == []


def test_200_chunk_filing_needs_about_25_calls(make_chunk):
    many = [make_chunk(chunk_id=f"chunk-{i}") for i in range(200)]
    assert len(en.batch_chunks(many)) == 25  # ~90% fewer calls than one-per-chunk


# ----------------------------------------------------------------------------- enrichment
async def test_enrich_merges_fields_by_chunk_id_and_preserves_order(chunks):
    g = FakeGenai(responder_for({"chunk-3": _meta("chunk-3", financial_metrics=["EPS", "net_income"],
                                                  is_comparative=False, currency_and_scale=None)}))
    out = await en.enrich_chunks(chunks[:8], client=g, limiter=None)
    assert [c.chunk_id for c in out] == [c.chunk_id for c in chunks[:8]]
    c3 = out[3]
    assert c3.financial_metrics == ["EPS", "net_income"] and c3.currency_and_scale is None
    assert c3.is_comparative is False
    c0 = out[0]
    assert (c0.financial_metrics, c0.fiscal_periods, c0.currency_and_scale, c0.is_comparative) == (
        ["revenue"], ["FY2024"], "USD in millions", True)


async def test_enrich_never_alters_core_chunk_content(chunks):
    out = await en.enrich_chunks(chunks[:8], client=FakeGenai(responder_for()))
    for before, after in zip(chunks[:8], out):
        assert (before.text, before.page_number, before.element_type, before.section_header) == (
            after.text, after.page_number, after.element_type, after.section_header)


async def test_one_call_per_batch_with_full_batch_context(chunks):
    g = FakeGenai(responder_for())
    await en.enrich_chunks(chunks, client=g)
    assert len(g.calls) == 3
    first = str(g.calls[0]["contents"])
    for c in chunks[:8]:
        assert c.chunk_id in first and c.text in first
    assert chunks[8].chunk_id not in first


async def test_gemini_call_configuration(chunks):
    g = FakeGenai(responder_for())
    await en.enrich_chunks(chunks[:6], client=g)
    call = g.calls[0]
    cfg = call["config"]
    assert call["model"] == "gemini-3.5-flash-lite"
    assert cfg.response_schema is en.BatchEnrichmentResponse
    assert cfg.response_mime_type == "application/json"
    assert cfg.system_instruction == en.ENRICHMENT_SYSTEM_PROMPT


async def test_limiter_acquired_once_per_call(chunks, limiter):
    g = FakeGenai(responder_for())
    await en.enrich_chunks(chunks, client=g, limiter=limiter)
    assert limiter.acquired == len(g.calls) == 3


async def test_default_client_and_limiter_come_from_clients_module(chunks, patch_clients):
    g = FakeGenai(responder_for())
    patch_clients(genai=g)
    out = await en.enrich_chunks(chunks[:6])
    assert len(g.calls) == 1 and out[0].financial_metrics == ["revenue"]


async def test_empty_input_makes_no_calls():
    g = FakeGenai(responder_for())
    assert await en.enrich_chunks([], client=g) == []
    assert g.calls == []


async def test_unknown_ids_ignored_and_missing_ids_left_untouched(chunks):
    g = FakeGenai(responder_for({"extra-zzz": _meta("extra-zzz")}, skip={"chunk-2"}))
    # inject the unknown id into the contents so the fake echoes it back
    chunks[0] = chunks[0].model_copy(update={"text": "mentions extra-zzz here"})
    out = await en.enrich_chunks(chunks[:6], client=g)
    assert out[2].financial_metrics == [] and out[2].currency_and_scale is None
    assert len(out) == 6 and all(c.chunk_id.startswith("chunk-") for c in out)


async def test_429_is_retried_with_backoff_then_succeeds(chunks, no_sleep, limiter):
    ok = responder_for()
    seq = iter([genai_error(429), genai_error(429), None])

    def respond(call):
        nxt = next(seq)
        return nxt if nxt else ok(call)

    g = FakeGenai(respond)
    out = await en.enrich_chunks(chunks[:6], client=g, limiter=limiter)
    assert len(g.calls) == 3 and limiter.acquired == 3  # every attempt is rate limited
    assert out[0].financial_metrics == ["revenue"]
    assert no_sleep and all(2 <= d <= 30 for d in no_sleep)


async def test_persistent_429_degrades_gracefully_after_five_attempts(chunks, no_sleep):
    g = FakeGenai(lambda call: genai_error(429))
    out = await en.enrich_chunks(chunks[:6], client=g)
    assert len(g.calls) == 5
    assert [c.chunk_id for c in out] == [c.chunk_id for c in chunks[:6]]
    assert all(c.financial_metrics == [] for c in out)


async def test_strict_mode_reraises(chunks, no_sleep):
    g = FakeGenai(lambda call: genai_error(429))
    with pytest.raises(Exception):
        await en.enrich_chunks(chunks[:6], client=g, strict=True)


async def test_one_bad_batch_does_not_poison_the_others(chunks, no_sleep):
    good = responder_for()
    state = {"n": 0}

    def respond(call):
        state["n"] += 1
        return "this is not json" if state["n"] == 1 else good(call)

    out = await en.enrich_chunks(chunks[:16], client=FakeGenai(respond))
    assert all(c.financial_metrics == [] for c in out[:8])
    assert all(c.financial_metrics == ["revenue"] for c in out[8:])


async def test_enrichment_is_idempotent(chunks):
    g = FakeGenai(responder_for())
    once = await en.enrich_chunks(chunks[:6], client=g)
    twice = await en.enrich_chunks(once, client=g)
    assert [c.model_dump() for c in once] == [c.model_dump() for c in twice]


# ----------------------------------------------------------------------------- 6.2 headings
PAGES = [
    (1, "# Part II\nintro\n## Item 8 Financial Statements\n### Note 3 — Revenue Recognition\nbody"),
    (2, "### Note 4 — Income Taxes\nbody\n## Item 9 Controls\n"),
    (3, "# Part III\n## Item 10 Directors\n"),
]


def test_extract_headings_levels_pages_and_breadcrumbs():
    hs = en.extract_headings(PAGES)
    assert [(h.level, h.text, h.page_number) for h in hs] == [
        (1, "Part II", 1), (2, "Item 8 Financial Statements", 1), (3, "Note 3 — Revenue Recognition", 1),
        (3, "Note 4 — Income Taxes", 2), (2, "Item 9 Controls", 2), (1, "Part III", 3),
        (2, "Item 10 Directors", 3)]
    crumbs = [h.breadcrumb for h in hs]
    assert crumbs[2] == "Part II > Item 8 Financial Statements > Note 3 — Revenue Recognition"
    assert crumbs[3] == "Part II > Item 8 Financial Statements > Note 4 — Income Taxes"  # sibling replaces
    assert crumbs[4] == "Part II > Item 9 Controls"
    assert crumbs[6] == "Part III > Item 10 Directors"
    assert crumbs[0] == "Part II"


def test_skipped_heading_levels_do_not_create_gaps():
    hs = en.extract_headings([(1, "# A\n### B")])
    assert hs[1].breadcrumb == "A > B"


def test_non_headings_are_ignored():
    hs = en.extract_headings([(1, "#hashtag not heading\nplain\n####### too deep\n## Real")])
    assert [h.text for h in hs] == ["Real"]


def test_extract_headings_empty():
    assert en.extract_headings([]) == [] and en.extract_headings([(1, "no headings")]) == []


# ----------------------------------------------------------------------------- section tree
def test_section_tree_structure_and_ids():
    secs = en.build_section_tree(en.extract_headings(PAGES), "doc-1")
    assert [s.header_text for s in secs] == [h.text for h in en.extract_headings(PAGES)]
    assert len({s.section_id for s in secs}) == len(secs)
    assert all(s.document_id == "doc-1" and s.element_type == "text" for s in secs)
    by_text = {s.header_text: s for s in secs}
    assert by_text["Part II"].parent_id is None and by_text["Part III"].parent_id is None
    assert by_text["Item 8 Financial Statements"].parent_id == by_text["Part II"].section_id
    assert by_text["Note 3 — Revenue Recognition"].parent_id == by_text["Item 8 Financial Statements"].section_id
    assert by_text["Note 4 — Income Taxes"].parent_id == by_text["Item 8 Financial Statements"].section_id
    assert by_text["Item 9 Controls"].parent_id == by_text["Part II"].section_id
    assert by_text["Note 3 — Revenue Recognition"].breadcrumb.endswith("Note 3 — Revenue Recognition")
    assert by_text["Note 4 — Income Taxes"].page_number == 2 and by_text["Part II"].level == 1


def test_section_ids_are_scoped_per_document():
    h = en.extract_headings(PAGES)
    a = {s.section_id for s in en.build_section_tree(h, "doc-1")}
    b = {s.section_id for s in en.build_section_tree(h, "doc-2")}
    assert a.isdisjoint(b)


def test_section_tree_empty():
    assert en.build_section_tree([], "d") == []


# ----------------------------------------------------------------------------- breadcrumbs
@pytest.fixture
def sections():
    return en.build_section_tree(en.extract_headings(PAGES), "doc-1")


def test_section_for_chunk_matches_header_and_page(make_chunk, sections):
    c = make_chunk(section_header="Note 4 — Income Taxes", page_number=2)
    assert en.section_for_chunk(c, sections).header_text == "Note 4 — Income Taxes"
    assert en.section_for_chunk(make_chunk(section_header="Nope", page_number=2), sections) is None
    assert en.section_for_chunk(make_chunk(section_header="", page_number=2), sections) is None


def test_section_for_chunk_prefers_latest_preceding_duplicate_heading(make_chunk):
    pages = [(1, "# Part I\n## Overview"), (5, "# Part II\n## Overview")]
    secs = en.build_section_tree(en.extract_headings(pages), "d")
    early = en.section_for_chunk(make_chunk(section_header="Overview", page_number=3), secs)
    late = en.section_for_chunk(make_chunk(section_header="Overview", page_number=7), secs)
    assert early.breadcrumb == "Part I > Overview" and late.breadcrumb == "Part II > Overview"


def test_assign_breadcrumbs(make_chunk, sections):
    cs = [
        make_chunk(section_header="Note 3 — Revenue Recognition", page_number=1),
        make_chunk(section_header="Unknown Heading", page_number=1),
        make_chunk(section_header="", page_number=1),
    ]
    out = en.assign_breadcrumbs(cs, sections)
    assert out[0].breadcrumb == "Part II > Item 8 Financial Statements > Note 3 — Revenue Recognition"
    assert out[1].breadcrumb == "Unknown Heading"
    assert out[2].breadcrumb == ""
    assert len(out) == 3


def test_breadcrumb_survives_payload_round_trip(make_chunk, sections):
    (c,) = en.assign_breadcrumbs([make_chunk(section_header="Item 9 Controls", page_number=2)], sections)
    assert json.loads(json.dumps(c.to_payload()))["breadcrumb"] == "Part II > Item 9 Controls"
