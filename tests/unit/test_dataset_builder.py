"""Phase 1.1 — golden dataset schema, validation, stratified smoke subset (edge cases)."""

from __future__ import annotations

import json
from collections import Counter

import pytest
from pydantic import ValidationError

from amorphnet.evaluation.dataset_builder import (
    FULL_SIZE, QUERY_TYPES, SMOKE_QUOTAS, SMOKE_SIZE, TARGET_DISTRIBUTION, GoldenQA,
    distribution, get_eval_subset, load_golden_dataset, select_smoke_subset, validate_dataset,
)


def _record(i: int = 0, qt: str = "factual_numeric", **kw):
    base = dict(
        question=f"What was the net revenue number {i} for the fiscal year?",
        ground_truth="It was $1 million.", company_name="Apple Inc.", fiscal_year=2025,
        filing_type="10-K", query_type=qt,
    )
    base.update(kw)
    return base


def _dataset(per_type: dict[str, int]):
    out, i = [], 0
    for qt, n in per_type.items():
        for _ in range(n):
            out.append(GoldenQA(**_record(i, qt)))
            i += 1
    return out


def test_constants_are_consistent_with_architecture_15_1():
    assert FULL_SIZE == 50 and SMOKE_SIZE == 10
    assert set(QUERY_TYPES) == set(TARGET_DISTRIBUTION) == set(SMOKE_QUOTAS)
    assert sum(TARGET_DISTRIBUTION.values()) == pytest.approx(1.0)
    assert sum(SMOKE_QUOTAS.values()) == SMOKE_SIZE


def test_real_golden_dataset_matches_distribution_and_has_unique_questions():
    items = load_golden_dataset()
    assert validate_dataset(items) == []
    counts = distribution(items)
    for qt, share in TARGET_DISTRIBUTION.items():
        assert abs(counts[qt] - share * 50) <= 1
    assert len({i.question.lower() for i in items}) == 50
    assert all(i.company_name == "Apple Inc." for i in items)


def test_golden_records_have_substantive_ground_truth_and_filters():
    for item in load_golden_dataset():
        assert len(item.ground_truth) > 20
        assert item.fiscal_year >= 2000
        assert item.filing_type in {"10-K", "10-Q", "20-F"}


def test_reference_chunks_default_to_empty_list_not_shared():
    a, b = GoldenQA(**_record(1)), GoldenQA(**_record(2))
    a.reference_chunks.append("x")
    assert b.reference_chunks == []


@pytest.mark.parametrize("field,value", [
    ("question", "short"), ("question", "   "), ("ground_truth", ""), ("company_name", " "),
    ("filing_type", ""), ("query_type", "opinion"), ("fiscal_year", "abc"),
])
def test_schema_rejects_bad_fields(field, value):
    with pytest.raises(ValidationError):
        GoldenQA(**_record(0, **{field: value}))


def test_schema_strips_whitespace():
    item = GoldenQA(**_record(0, company_name="  Apple Inc.  "))
    assert item.company_name == "Apple Inc."


def test_filters_property_matches_query_api_contract():
    assert GoldenQA(**_record()).filters == {
        "company_name": "Apple Inc.", "fiscal_year": 2025, "filing_type": "10-K"}


def test_load_missing_file_and_non_array(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_golden_dataset(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"a": 1}))
    with pytest.raises(ValueError):
        load_golden_dataset(bad)


def test_load_rejects_invalid_record(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([_record(0, query_type="nonsense")]))
    with pytest.raises(ValidationError):
        load_golden_dataset(p)


def test_distribution_always_lists_all_types():
    assert distribution([]) == {qt: 0 for qt in QUERY_TYPES}


def test_validate_flags_distribution_skew_and_size():
    skewed = _dataset({"factual_numeric": 50, "trend_comparative": 0, "semantic_risk": 0,
                       "cross_reference": 0})
    problems = validate_dataset(skewed)
    assert any("factual_numeric" in p for p in problems)
    assert validate_dataset(skewed[:10])  # wrong size


def test_validate_duplicate_detection_is_case_insensitive():
    items = _dataset({"factual_numeric": 15, "trend_comparative": 13, "semantic_risk": 12,
                      "cross_reference": 10})
    items[1] = items[0].model_copy(update={"question": items[0].question.upper()})
    assert any("Duplicate" in p for p in validate_dataset(items))


def test_smoke_subset_spans_each_type_evenly_and_keeps_dataset_order():
    items = load_golden_dataset()
    subset = select_smoke_subset(items)
    index = {id(i): n for n, i in enumerate(items)}
    positions = [index[id(s)] for s in subset]
    assert positions == sorted(positions)
    # Not simply "first N of each type": at least one pick is from the second half of its type.
    for qt in QUERY_TYPES:
        pool = [n for n, i in enumerate(items) if i.query_type == qt]
        picks = [p for p in positions if p in pool]
        assert picks[-1] > pool[len(pool) // 2 - 1]


def test_smoke_subset_raises_when_type_too_small():
    items = _dataset({"factual_numeric": 3, "trend_comparative": 3, "semantic_risk": 1,
                      "cross_reference": 2})
    with pytest.raises(ValueError, match="semantic_risk"):
        select_smoke_subset(items)


def test_smoke_subset_custom_quotas():
    items = load_golden_dataset()
    q = {"factual_numeric": 1, "trend_comparative": 1, "semantic_risk": 1, "cross_reference": 1}
    assert Counter(i.query_type for i in select_smoke_subset(items, q)) == Counter(q)


def test_get_eval_subset_modes_and_unknown(tmp_path):
    assert len(get_eval_subset("smoke")) == 10
    assert len(get_eval_subset("full")) == 50
    with pytest.raises(ValueError):
        get_eval_subset("medium")  # type: ignore[arg-type]
