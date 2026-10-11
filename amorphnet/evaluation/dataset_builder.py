"""Golden evaluation dataset builder and validator.

Phase 1.1 — Golden Evaluation Dataset
Utilities to curate, format, validate, and sample smoke-eval (10 QA) subsets from golden QA data.

The golden dataset (``evaluation/golden_dataset.json``) is the single source of truth
consumed by the Tier 1 baseline evaluation (Phase 3) and the full ablation benchmark
(Phase 14). Each record carries the filter metadata (``company_name``, ``fiscal_year``,
``filing_type``) that the retrieval layer uses as Qdrant payload filters, so evaluated
pipelines receive exactly the same filter contract as the production ``/query`` API.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DATASET_PATH = PROJECT_ROOT / "evaluation" / "golden_dataset.json"

QueryType = Literal["factual_numeric", "trend_comparative", "semantic_risk", "cross_reference"]
QUERY_TYPES: tuple[str, ...] = (
    "factual_numeric",
    "trend_comparative",
    "semantic_risk",
    "cross_reference",
)

# Section 15.1 target distribution: 30 / 25 / 25 / 20 %
TARGET_DISTRIBUTION: dict[str, float] = {
    "factual_numeric": 0.30,
    "trend_comparative": 0.25,
    "semantic_risk": 0.25,
    "cross_reference": 0.20,
}

FULL_SIZE = 50
SMOKE_SIZE = 10

# Smoke subset quotas (10 items). 30/25/25/20 % of 10 = 3 / 2.5 / 2.5 / 2; the two
# half-items cannot be split, so the extra slot goes to ``trend_comparative`` (which has
# one more item than ``semantic_risk`` in the full set). Largest deviation: 5 percentage points.
SMOKE_QUOTAS: dict[str, int] = {
    "factual_numeric": 3,
    "trend_comparative": 3,
    "semantic_risk": 2,
    "cross_reference": 2,
}


class GoldenQA(BaseModel):
    """A single curated golden QA record."""

    question: str = Field(min_length=10)
    ground_truth: str = Field(min_length=1)
    company_name: str = Field(min_length=1)
    fiscal_year: int
    filing_type: str = Field(min_length=1)
    query_type: QueryType
    # chunk_ids of gold source chunks. Chunk IDs are UUIDs minted at ingestion time
    # (Phase 5), so this stays empty until the filings are chunked and resolved.
    reference_chunks: list[str] = Field(default_factory=list)

    @field_validator("question", "ground_truth", "company_name", "filing_type")
    @classmethod
    def _strip_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @property
    def filters(self) -> dict[str, object]:
        """Payload-filter dict matching the ``/query`` API ``filters`` contract."""
        return {
            "company_name": self.company_name,
            "fiscal_year": self.fiscal_year,
            "filing_type": self.filing_type,
        }


def load_golden_dataset(path: Path | str = GOLDEN_DATASET_PATH) -> list[GoldenQA]:
    """Load and schema-validate the golden dataset from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        raise ValueError("Golden dataset must be a JSON array of QA records.")
    return [GoldenQA.model_validate(item) for item in raw]


def distribution(items: list[GoldenQA]) -> dict[str, int]:
    """Count items per query type (all four types always present)."""
    counts = Counter(item.query_type for item in items)
    return {qt: counts.get(qt, 0) for qt in QUERY_TYPES}


def validate_dataset(
    items: list[GoldenQA],
    expected_size: int = FULL_SIZE,
    tolerance: int = 1,
) -> list[str]:
    """Return a list of validation problems (empty list means the dataset is valid).

    Per-type counts may deviate from the exact target by ``tolerance`` items, since
    25% of 50 (12.5) is not an integer.
    """
    problems: list[str] = []

    if len(items) != expected_size:
        problems.append(f"Expected {expected_size} QA pairs, found {len(items)}.")

    seen: set[str] = set()
    for idx, item in enumerate(items):
        key = item.question.lower()
        if key in seen:
            problems.append(f"Duplicate question at index {idx}: {item.question[:60]!r}")
        seen.add(key)

    counts = distribution(items)
    for qt, share in TARGET_DISTRIBUTION.items():
        target = share * len(items)
        if abs(counts[qt] - target) > tolerance:
            problems.append(
                f"Query type '{qt}' has {counts[qt]} items; target ~{target:g} (±{tolerance})."
            )
    return problems


def select_smoke_subset(
    items: list[GoldenQA],
    quotas: dict[str, int] | None = None,
) -> list[GoldenQA]:
    """Deterministically select the stratified 10-question smoke subset.

    Within each query type, items are picked at evenly spaced positions so the subset
    spans the whole breadth of the type (rather than just its first few items), and the
    selection is reproducible across runs. Result preserves original dataset order.
    """
    quotas = quotas or SMOKE_QUOTAS
    by_type: dict[str, list[tuple[int, GoldenQA]]] = defaultdict(list)
    for idx, item in enumerate(items):
        by_type[item.query_type].append((idx, item))

    chosen: list[tuple[int, GoldenQA]] = []
    for qt, quota in quotas.items():
        pool = by_type.get(qt, [])
        if len(pool) < quota:
            raise ValueError(f"Query type '{qt}' has {len(pool)} items; smoke quota needs {quota}.")
        # Evenly spaced picks: centre of each of `quota` equal buckets.
        picks = {int((i + 0.5) * len(pool) / quota) for i in range(quota)}
        chosen.extend(pool[p] for p in sorted(picks))

    chosen.sort(key=lambda pair: pair[0])
    return [item for _, item in chosen]


def get_eval_subset(
    mode: Literal["smoke", "full"],
    path: Path | str = GOLDEN_DATASET_PATH,
) -> list[GoldenQA]:
    """Load the dataset for an evaluation mode (``smoke`` = 10 items, ``full`` = all 50)."""
    items = load_golden_dataset(path)
    if mode == "smoke":
        return select_smoke_subset(items)
    if mode == "full":
        return items
    raise ValueError(f"Unknown evaluation mode: {mode!r}")


def main() -> int:  # pragma: no cover - convenience CLI
    items = load_golden_dataset()
    problems = validate_dataset(items)
    print(f"Golden dataset: {len(items)} items; distribution={distribution(items)}")
    print(f"Smoke subset distribution={distribution(select_smoke_subset(items))}")
    for problem in problems:
        print(f"PROBLEM: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
