"""Phase 7.3 / 7.4 — ``amorphnet/ingestion/neo4j_indexer.py`` (knowledge graph + cross-references).

Contract:
  ``find_cross_references(text) -> list[str]`` — normalised, de-duplicated (first-seen order) targets for
      ``See Note N`` / ``as described in Note N`` / ``Refer to <Word> N`` / ``discussed in Item N`` /
      ``as shown in [Tt]able N`` (lead-ins are case-insensitive, e.g. ``(see Note 7)``), e.g.
      ``["Note 7", "Item 7", "Table 3"]``.
  ``resolve_cross_references(chunks) -> list[tuple[source_chunk_id, target_chunk_id]]``
      A target label (``"Note 7"``) matches chunks whose ``section_header`` / ``breadcrumb`` /
      ``table_caption`` contains it as a whole token (``Note 7`` never matches ``Note 70``).
      Self references and references into the source's own section are dropped; edges are unique.
  ``CONSTRAINT_STATEMENTS`` — 4 Cypher ``CREATE CONSTRAINT ... IF NOT EXISTS ... IS UNIQUE`` statements
      for ``Document.document_id``, ``Section.section_id``, ``Chunk.chunk_id``, ``FinancialEntity.name``.
  ``async ensure_constraints(driver)`` — runs each statement via ``session.run``.
  ``build_graph_payload(document: dict, sections: list[SectionNode], chunks: list[Chunk]) -> GraphPayload``
      ``.document``  {document_id, company_name, filing_type, fiscal_year}
      ``.sections``  [{section_id, document_id, header_text, breadcrumb, page_number, element_type}]
      ``.chunks``    [{chunk_id, section_id, element_type, page_number, token_count}] (section_id ""
                     when the chunk matches no section)
      ``.entities``  unique [{name, entity_type}] — ``financial_metrics`` => "metric",
                     ``fiscal_periods`` => "period"
      ``.edges``     dict of relationship type -> list[(from_id, to_id)] with keys HAS_SECTION (document ->
                     ROOT sections), HAS_CHILD_SECTION (parent -> child), HAS_CHUNK (section -> chunk),
                     MENTIONS (chunk -> entity name), NEXT_SECTION (document-order neighbours),
                     FOOTNOTE_OF (footnote chunk -> parent chunk), REFERENCES (cross-reference edges)
  ``async write_graph(payload, driver, *, batch_size=500)`` — idempotent ``MERGE``-based writes using
      ``UNWIND $rows``: every write query receives ``rows=[...]`` (<= ``batch_size`` items); data is
      always passed as parameters (never interpolated into Cypher).
  ``async index_document_graph(document, sections, chunks, *, driver) -> GraphPayload`` —
      ``ensure_constraints`` then ``build_graph_payload`` then ``write_graph``.
"""

from __future__ import annotations

import pytest

from tests.support import FakeNeo4j, require

nj = require("amorphnet.ingestion.neo4j_indexer", "find_cross_references", "resolve_cross_references",
             "CONSTRAINT_STATEMENTS", "ensure_constraints", "build_graph_payload", "write_graph",
             "index_document_graph")
en = require("amorphnet.ingestion.metadata_enricher", "extract_headings", "build_section_tree")

RELATIONSHIPS = ("HAS_SECTION", "HAS_CHILD_SECTION", "HAS_CHUNK", "MENTIONS", "NEXT_SECTION",
                 "FOOTNOTE_OF", "REFERENCES")


# ----------------------------------------------------------------------------- 7.4 patterns
@pytest.mark.parametrize("text,expected", [
    ("Details are in the notes. See Note 7 for more.", ["Note 7"]),
    ("Revenue is explained as described in Note 12.", ["Note 12"]),
    ("Refer to Note 3 for lease terms.", ["Note 3"]),
    ("Refer to Part 2 for governance.", ["Part 2"]),
    ("Risks are discussed in Item 1 and again in Item 7.", ["Item 1", "Item 7"]),
    ("as shown in Table 3 below", ["Table 3"]),
    ("as shown in table 3 below", ["Table 3"]),
    ("(see Note 7)", ["Note 7"]),
    ("Taxes: see Note 70 for others.", ["Note 70"]),
])
def test_find_cross_references_positive(text, expected):
    assert nj.find_cross_references(text) == expected


@pytest.mark.parametrize("text", [
    "", "Note that revenue grew 5%.", "See Note for details.", "Notes 7 are long.",
    "The table shows results.", "discussed in the Item.", "We refer to our policies.",
])
def test_find_cross_references_negative(text):
    assert nj.find_cross_references(text) == []


def test_find_cross_references_dedupes_and_keeps_first_seen_order():
    text = "See Note 7. Also, as shown in Table 3, and again see Note 7; discussed in Item 7."
    assert nj.find_cross_references(text) == ["Note 7", "Table 3", "Item 7"]


# ----------------------------------------------------------------------------- fixtures
PAGES = [(1, "# Part II\n## Item 8\n### Note 3 — Revenue Recognition\n"), (2, "### Note 7 — Income Taxes\n")]


@pytest.fixture
def world(make_chunk):
    sections = en.build_section_tree(en.extract_headings(PAGES), "doc-1")
    sec = {s.header_text: s for s in sections}
    a = make_chunk(chunk_id="c-a", section_header="Note 3 — Revenue Recognition", page_number=1,
                   text="Revenue grew 6%. See Note 7 for taxes.", financial_metrics=["revenue", "EPS"],
                   fiscal_periods=["FY2024"], token_count=11)
    foot = make_chunk(chunk_id="c-foot", section_header="Note 3 — Revenue Recognition", page_number=1,
                      element_type="footnote", parent_anchor="(1)", parent_chunk_id="c-a",
                      text="(1) Includes FX.", token_count=3)
    tax = make_chunk(chunk_id="c-tax", section_header="Note 7 — Income Taxes", page_number=2,
                     text="Income taxes were $1.", financial_metrics=["revenue"], token_count=4)
    tab = make_chunk(chunk_id="c-tab", section_header="Note 7 — Income Taxes", page_number=2,
                     element_type="table", table_caption="Table 3 effective tax rate", text="| a |",
                     token_count=2)
    orphan = make_chunk(chunk_id="c-orphan", section_header="", page_number=1, text="Preamble.")
    document = dict(document_id="doc-1", company_name="Apple Inc.", filing_type="10-K", fiscal_year=2025,
                    filename="ignored.pdf", sha256_hash="ignored")
    return dict(sections=sections, sec=sec, chunks=[a, foot, tax, tab, orphan], document=document)


def edges(payload, rel):
    return {tuple(e) for e in payload.edges[rel]}


# ----------------------------------------------------------------------------- resolution
def test_resolve_cross_references_links_to_every_chunk_of_the_target_section(world):
    pairs = set(nj.resolve_cross_references(world["chunks"]))
    assert pairs == {("c-a", "c-tax"), ("c-a", "c-tab")}


def test_resolution_drops_self_and_same_section_references(make_chunk):
    own = make_chunk(chunk_id="x", section_header="Note 3", text="As described in Note 3 above.")
    sibling = make_chunk(chunk_id="y", section_header="Note 3", text="Sibling text.")
    assert nj.resolve_cross_references([own, sibling]) == []


def test_resolution_uses_whole_token_matching(make_chunk):
    src = make_chunk(chunk_id="s", section_header="Item 7", text="See Note 7.")
    wrong = make_chunk(chunk_id="w", section_header="Note 70 — Other")
    right = make_chunk(chunk_id="r", section_header="Note 7 — Taxes")
    assert nj.resolve_cross_references([src, wrong, right]) == [("s", "r")]


def test_resolution_matches_breadcrumb_and_table_caption(make_chunk):
    src = make_chunk(chunk_id="s", section_header="Intro", text="as shown in Table 3.")
    by_caption = make_chunk(chunk_id="t", section_header="Other", table_caption="Table 3: Net sales")
    by_crumb = make_chunk(chunk_id="b", section_header="Sub", breadcrumb="Part I > Table 3 data")
    assert set(nj.resolve_cross_references([src, by_caption, by_crumb])) == {("s", "t"), ("s", "b")}


def test_unresolvable_references_create_no_edges_and_edges_are_unique(make_chunk):
    src = make_chunk(chunk_id="s", section_header="A", text="See Note 9. See Note 9. See Note 1.")
    tgt = make_chunk(chunk_id="t", section_header="Note 1")
    out = nj.resolve_cross_references([src, tgt])
    assert out == [("s", "t")]


# ----------------------------------------------------------------------------- constraints
def test_constraint_statements_cover_all_four_node_types():
    stmts = list(nj.CONSTRAINT_STATEMENTS)
    assert len(stmts) == 4
    for s in stmts:
        assert "CREATE CONSTRAINT" in s.upper() and "IF NOT EXISTS" in s.upper() and "IS UNIQUE" in s.upper()
    joined = "\n".join(stmts)
    for label, prop in (("Document", "document_id"), ("Section", "section_id"),
                        ("Chunk", "chunk_id"), ("FinancialEntity", "name")):
        assert f":{label})" in joined and f".{prop}" in joined


async def test_ensure_constraints_runs_every_statement():
    d = FakeNeo4j()
    await nj.ensure_constraints(d)
    assert [q for q, _ in d.queries] == list(nj.CONSTRAINT_STATEMENTS)


# ----------------------------------------------------------------------------- payload
def test_payload_nodes(world):
    p = nj.build_graph_payload(world["document"], world["sections"], world["chunks"])
    assert p.document == {"document_id": "doc-1", "company_name": "Apple Inc.", "filing_type": "10-K",
                          "fiscal_year": 2025}
    assert [s["header_text"] for s in p.sections] == ["Part II", "Item 8", "Note 3 — Revenue Recognition",
                                                      "Note 7 — Income Taxes"]
    s0 = p.sections[3]
    assert set(s0) == {"section_id", "document_id", "header_text", "breadcrumb", "page_number",
                       "element_type"}
    assert s0["breadcrumb"] == "Part II > Item 8 > Note 7 — Income Taxes" and s0["page_number"] == 2
    cmap = {c["chunk_id"]: c for c in p.chunks}
    assert set(cmap) == {"c-a", "c-foot", "c-tax", "c-tab", "c-orphan"}
    assert set(cmap["c-a"]) == {"chunk_id", "section_id", "element_type", "page_number", "token_count"}
    assert cmap["c-a"]["section_id"] == world["sec"]["Note 3 — Revenue Recognition"].section_id
    assert cmap["c-a"]["token_count"] == 11 and cmap["c-tab"]["element_type"] == "table"
    assert cmap["c-orphan"]["section_id"] == ""


def test_payload_entities_unique_with_types(world):
    p = nj.build_graph_payload(world["document"], world["sections"], world["chunks"])
    assert sorted((e["name"], e["entity_type"]) for e in p.entities) == [
        ("EPS", "metric"), ("FY2024", "period"), ("revenue", "metric")]


def test_payload_relationships(world):
    p = nj.build_graph_payload(world["document"], world["sections"], world["chunks"])
    assert set(p.edges) == set(RELATIONSHIPS)
    sec = {k: v.section_id for k, v in world["sec"].items()}
    part, item = sec["Part II"], sec["Item 8"]
    n3, n7 = sec["Note 3 — Revenue Recognition"], sec["Note 7 — Income Taxes"]
    assert edges(p, "HAS_SECTION") == {("doc-1", part)}  # root sections only
    assert edges(p, "HAS_CHILD_SECTION") == {(part, item), (item, n3), (item, n7)}
    assert edges(p, "HAS_CHUNK") == {(n3, "c-a"), (n3, "c-foot"), (n7, "c-tax"), (n7, "c-tab")}
    assert edges(p, "NEXT_SECTION") == {(part, item), (item, n3), (n3, n7)}
    assert edges(p, "MENTIONS") == {("c-a", "revenue"), ("c-a", "EPS"), ("c-a", "FY2024"),
                                    ("c-tax", "revenue")}
    assert edges(p, "FOOTNOTE_OF") == {("c-foot", "c-a")}
    assert edges(p, "REFERENCES") == {("c-a", "c-tax"), ("c-a", "c-tab")}


def test_payload_empty_inputs_are_safe():
    p = nj.build_graph_payload(dict(document_id="d", company_name="x", filing_type="10-K", fiscal_year=2024),
                               [], [])
    assert p.sections == [] and p.chunks == [] and p.entities == []
    assert all(p.edges[r] == [] for r in RELATIONSHIPS)


def test_footnote_without_parent_creates_no_edge(make_chunk):
    f = make_chunk(chunk_id="f", element_type="footnote", parent_anchor="(2)", parent_chunk_id="")
    p = nj.build_graph_payload(dict(document_id="d", company_name="x", filing_type="10-K", fiscal_year=2024),
                               [], [f])
    assert p.edges["FOOTNOTE_OF"] == []


# ----------------------------------------------------------------------------- write_graph
def _expected_rows(p):
    return 1 + len(p.sections) + len(p.chunks) + len(p.entities) + sum(len(v) for v in p.edges.values())


async def test_write_graph_uses_merge_unwind_and_covers_every_node_and_relationship(world):
    p = nj.build_graph_payload(world["document"], world["sections"], world["chunks"])
    d = FakeNeo4j()
    await nj.write_graph(p, d)
    cy = d.cypher()
    assert "MERGE" in cy and "UNWIND" in cy
    for label in ("Document", "Section", "Chunk", "FinancialEntity"):
        assert f":{label}" in cy
    for rel in RELATIONSHIPS:
        assert f":{rel}" in cy
    assert "CREATE (" not in cy  # re-ingestion must not duplicate nodes


async def test_write_graph_writes_every_row_exactly_once(world):
    p = nj.build_graph_payload(world["document"], world["sections"], world["chunks"])
    d = FakeNeo4j()
    await nj.write_graph(p, d)
    assert sum(len(params["rows"]) for _, params in d.queries) == _expected_rows(p)


async def test_write_graph_batches_rows(world):
    p = nj.build_graph_payload(world["document"], world["sections"], world["chunks"])
    d = FakeNeo4j()
    await nj.write_graph(p, d, batch_size=3)
    assert all(len(params["rows"]) <= 3 for _, params in d.queries)
    assert sum(len(params["rows"]) for _, params in d.queries) == _expected_rows(p)


async def test_write_graph_parameterises_data_instead_of_interpolating(world):
    p = nj.build_graph_payload(world["document"], world["sections"], world["chunks"])
    d = FakeNeo4j()
    await nj.write_graph(p, d)
    for needle in ("c-a", "Apple Inc.", "doc-1", "revenue"):
        assert needle not in d.cypher()


async def test_write_graph_creates_nodes_before_relationships(world):
    p = nj.build_graph_payload(world["document"], world["sections"], world["chunks"])
    d = FakeNeo4j()
    await nj.write_graph(p, d)
    texts = [q for q, _ in d.queries]
    first_rel = next(i for i, q in enumerate(texts) if "MATCH" in q.upper() and "-[" in q)
    last_node_only = max(i for i, q in enumerate(texts) if "-[" not in q)
    assert last_node_only < first_rel


async def test_write_graph_empty_payload_does_nothing_harmful():
    p = nj.build_graph_payload(dict(document_id="d", company_name="x", filing_type="10-K", fiscal_year=2024),
                               [], [])
    d = FakeNeo4j()
    await nj.write_graph(p, d)  # only the Document node row is written
    assert sum(len(params["rows"]) for _, params in d.queries) == 1


async def test_index_document_graph_orchestrates_constraints_then_writes(world):
    d = FakeNeo4j()
    p = await nj.index_document_graph(world["document"], world["sections"], world["chunks"], driver=d)
    texts = [q for q, _ in d.queries]
    assert texts[:4] == list(nj.CONSTRAINT_STATEMENTS)
    assert any("MERGE" in t for t in texts[4:])
    assert p.document["document_id"] == "doc-1"


async def test_neo4j_errors_propagate(world):
    def boom(q, params):
        raise RuntimeError("aura down")

    with pytest.raises(RuntimeError):
        await nj.index_document_graph(world["document"], world["sections"], world["chunks"],
                                      driver=FakeNeo4j(boom))
