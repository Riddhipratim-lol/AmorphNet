"""Phase 5.1 — ``amorphnet/ingestion/chunk_schema.py``.

Contract:
  ``ELEMENT_TYPES == ("text", "table", "chart", "footnote")``
  ``Chunk`` (pydantic v2).  Required: ``document_id, company_name, filing_type, fiscal_year,
  page_number (>=1), section_header, element_type, text``.
  Defaults: ``chunk_id`` = fresh UUID4 str, ``ticker=""``, ``fiscal_quarter="Annual"``,
  ``table_caption=""``, ``parent_anchor=""``, ``parent_chunk_id=""``, ``chunk_index=0``,
  ``token_count=0``, ``char_offset_start=0``, ``char_offset_end=0``, and the Phase 6 enrichment
  fields ``financial_metrics=[]``, ``fiscal_periods=[]``, ``currency_and_scale=None``,
  ``is_comparative=False``, ``breadcrumb=""``; runtime-only retrieval fields ``score=0.0`` and
  ``source=""`` (``"qdrant"`` | ``"graph"``).
  ``chunk.to_payload() -> dict`` — JSON-serialisable Qdrant payload: every field EXCEPT ``score``
  and ``source`` (``text`` IS included so retrieval can return it).
  ``Chunk.from_payload(payload, *, score=0.0, source="") -> Chunk`` inverse of ``to_payload``.
"""

from __future__ import annotations

import json
import uuid

import pytest
from pydantic import ValidationError

from tests.support import require

schema = require("amorphnet.ingestion.chunk_schema", "Chunk", "ELEMENT_TYPES")
Chunk = schema.Chunk

REQUIRED = dict(document_id="d", company_name="Apple Inc.", filing_type="10-K", fiscal_year=2024,
                page_number=3, section_header="Item 7", element_type="text", text="hello")

ARCH_METADATA_FIELDS = {
    "chunk_id", "document_id", "company_name", "ticker", "filing_type", "fiscal_year",
    "fiscal_quarter", "page_number", "section_header", "element_type", "table_caption",
    "parent_anchor", "chunk_index", "token_count", "char_offset_start", "char_offset_end",
}


def test_element_types():
    assert tuple(schema.ELEMENT_TYPES) == ("text", "table", "chart", "footnote")


def test_model_contains_every_architecture_metadata_field():
    assert ARCH_METADATA_FIELDS | {"text"} <= set(Chunk.model_fields)


def test_defaults():
    c = Chunk(**REQUIRED)
    uuid.UUID(c.chunk_id)
    assert (c.ticker, c.fiscal_quarter, c.table_caption, c.parent_anchor) == ("", "Annual", "", "")
    assert (c.chunk_index, c.token_count, c.char_offset_start, c.char_offset_end) == (0, 0, 0, 0)
    assert c.financial_metrics == [] and c.fiscal_periods == [] and c.currency_and_scale is None
    assert c.is_comparative is False and c.breadcrumb == "" and c.parent_chunk_id == ""
    assert c.score == 0.0 and c.source == ""


def test_chunk_ids_are_unique_and_lists_not_shared():
    a, b = Chunk(**REQUIRED), Chunk(**REQUIRED)
    assert a.chunk_id != b.chunk_id
    a.financial_metrics.append("revenue")
    assert b.financial_metrics == []


@pytest.mark.parametrize("bad", ["image", "TEXT", "", "paragraph"])
def test_element_type_is_validated(bad):
    with pytest.raises(ValidationError):
        Chunk(**{**REQUIRED, "element_type": bad})


@pytest.mark.parametrize("element_type", ["text", "table", "chart", "footnote"])
def test_all_element_types_accepted(element_type):
    assert Chunk(**{**REQUIRED, "element_type": element_type}).element_type == element_type


@pytest.mark.parametrize("field,value", [("page_number", 0), ("page_number", -1),
                                         ("fiscal_year", "twenty"), ("text", None)])
def test_invalid_values_rejected(field, value):
    with pytest.raises(ValidationError):
        Chunk(**{**REQUIRED, field: value})


@pytest.mark.parametrize("missing", sorted(REQUIRED))
def test_required_fields(missing):
    data = {k: v for k, v in REQUIRED.items() if k != missing}
    with pytest.raises(ValidationError):
        Chunk(**data)


def test_payload_excludes_runtime_fields_includes_text_and_is_json_safe():
    c = Chunk(**REQUIRED, score=0.9, source="qdrant", financial_metrics=["revenue"])
    p = c.to_payload()
    assert "score" not in p and "source" not in p
    assert p["text"] == "hello" and ARCH_METADATA_FIELDS <= set(p)
    assert json.loads(json.dumps(p)) == p


def test_payload_round_trip():
    c = Chunk(**REQUIRED, table_caption="cap", parent_anchor="(1)", parent_chunk_id="x",
              financial_metrics=["EPS"], fiscal_periods=["FY2024"], currency_and_scale="USD in millions",
              is_comparative=True, breadcrumb="Part II > Item 7")
    back = Chunk.from_payload(c.to_payload(), score=0.42, source="graph")
    assert back.to_payload() == c.to_payload()
    assert back.score == 0.42 and back.source == "graph" and back.chunk_id == c.chunk_id


def test_from_payload_tolerates_missing_optional_fields():
    payload = {k: v for k, v in Chunk(**REQUIRED).to_payload().items()
               if k in set(REQUIRED) | {"chunk_id"}}
    assert Chunk.from_payload(payload).text == "hello"
