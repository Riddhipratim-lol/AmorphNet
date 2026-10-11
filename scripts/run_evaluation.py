#!/usr/bin/env python3
"""AmorphNet RAGAS Evaluation CLI.

Phase 1.3, 14.2 — Evaluation Runner
Executes evaluation on golden_dataset.json in --mode smoke (10 QA) or --mode full (50 QA).

Usage:
    python scripts/run_evaluation.py --mode smoke
    python scripts/run_evaluation.py --pipeline baseline --mode smoke
    python scripts/run_evaluation.py --pipeline oracle --mode smoke       # harness self-check
    python scripts/run_evaluation.py --from-file outputs.json --pipeline baseline
    python scripts/run_evaluation.py --mode full --dry-run                # no API calls

Pipelines (``--pipeline``):
    oracle    Harness validation only: answer = ground truth, context = ground truth.
              Scores should be near-perfect; low scores indicate a harness problem.
    baseline  Tier 1 pipeline (Phase 2). Needs ``amorphnet.generation.baseline_generator.
              answer_question(question, filters) -> {"answer": str, "contexts": list[str]}``.
    full      Tier 2 LangGraph pipeline (Phase 12): ``amorphnet.graph.langgraph_pipeline.
              amorphnet_pipeline``.

``--from-file`` evaluates pre-generated outputs: a JSON array (aligned with the selected
subset order, or keyed by ``question``) of ``{"question"?, "answer", "contexts"}``.

Results are merged into ``evaluation/ablation_results.json`` as
``{<pipeline>: {<mode>: report}}`` so smoke and full runs never overwrite each other.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from amorphnet.evaluation.dataset_builder import (  # noqa: E402
    GOLDEN_DATASET_PATH,
    GoldenQA,
    distribution,
    get_eval_subset,
    load_golden_dataset,
    validate_dataset,
)
from amorphnet.evaluation.ragas_runner import (  # noqa: E402
    EvalSample,
    PipelineFn,
    PipelineOutput,
    RagasReport,
    evaluate_samples,
    generate_samples,
)

logger = logging.getLogger("run_evaluation")

RESULTS_PATH = PROJECT_ROOT / "evaluation" / "ablation_results.json"
PIPELINES = ("oracle", "baseline", "full")
LLM_CALLS_PER_QUESTION = 5  # ~1 judge-call budget unit per metric per question (Section 15.1)


# -----------------------------------------------------------------------------
# Pipeline adapters
# -----------------------------------------------------------------------------
def _oracle_pipeline(items: list[GoldenQA]) -> PipelineFn:
    truth = {item.question: item.ground_truth for item in items}

    async def run(question: str, filters: Mapping[str, Any]) -> PipelineOutput:
        return PipelineOutput(answer=truth[question], contexts=[truth[question]])

    return run


def _baseline_pipeline() -> PipelineFn:
    try:
        module = importlib.import_module("amorphnet.generation.baseline_generator")
        fn = module.answer_question
    except (ImportError, AttributeError) as err:
        raise SystemExit(
            "Baseline pipeline is not available yet (Phase 2). Expected "
            "amorphnet.generation.baseline_generator.answer_question(question, filters)."
        ) from err

    async def run(question: str, filters: Mapping[str, Any]) -> PipelineOutput:
        out = await fn(question, dict(filters))
        return PipelineOutput(answer=out["answer"], contexts=list(out["contexts"]))

    return run


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, Mapping):
        return str(chunk.get("text", ""))
    return str(getattr(chunk, "text", ""))


def _full_pipeline() -> PipelineFn:
    try:
        module = importlib.import_module("amorphnet.graph.langgraph_pipeline")
        app = module.amorphnet_pipeline
    except (ImportError, AttributeError) as err:
        raise SystemExit(
            "Full pipeline is not available yet (Phase 12). Expected "
            "amorphnet.graph.langgraph_pipeline.amorphnet_pipeline."
        ) from err

    async def run(question: str, filters: Mapping[str, Any]) -> PipelineOutput:
        state = await app.ainvoke({"original_query": question, "filters": dict(filters)})
        return PipelineOutput(
            answer=state.get("answer", ""),
            contexts=[_chunk_text(c) for c in state.get("reranked_chunks", [])],
        )

    return run


def _from_file_samples(path: Path, items: list[GoldenQA]) -> list[EvalSample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit("--from-file must contain a JSON array.")
    by_question = {r["question"]: r for r in raw if isinstance(r, dict) and "question" in r}
    samples: list[EvalSample] = []
    for idx, item in enumerate(items):
        rec = by_question.get(item.question) or (raw[idx] if idx < len(raw) else None)
        if rec is None:
            raise SystemExit(f"No output found in {path} for question: {item.question[:70]!r}")
        samples.append(
            EvalSample(
                question=item.question,
                answer=str(rec.get("answer", "")),
                contexts=[str(c) for c in rec.get("contexts", [])],
                ground_truth=item.ground_truth,
                query_type=item.query_type,
            )
        )
    return samples


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
def print_summary(report: RagasReport) -> None:
    print(f"\n=== RAGAS results: pipeline={report.pipeline} mode={report.mode} "
          f"(n={report.num_samples}) ===")
    print(f"{'Metric':<22}{'Score':>8}{'Target':>8}  Status")
    for metric, target in report.targets.items():
        score = report.metrics[metric]
        met = report.targets_met[metric]
        shown = "n/a" if score is None else f"{score:.3f}"
        status = "n/a" if met is None else ("PASS" if met else "below target")
        print(f"{metric:<22}{shown:>8}{target:>8.2f}  {status}")
    print("\nBy query type:")
    for qt, scores in sorted(report.by_query_type.items()):
        cells = "  ".join(
            f"{m[:9]}={'n/a' if v is None else format(v, '.2f')}" for m, v in scores.items()
        )
        print(f"  {qt:<18}{cells}")


def persist_report(report: RagasReport, path: Path = RESULTS_PATH) -> None:
    """Merge ``report`` into ablation_results.json under ``[pipeline][mode]``."""
    existing: dict[str, Any] = {}
    if path.exists() and path.read_text(encoding="utf-8").strip():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.setdefault(report.pipeline, {})[report.mode] = report.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nResults saved to {path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AmorphNet RAGAS evaluation.")
    parser.add_argument("--mode", choices=("smoke", "full"), default=None,
                        help="smoke = 10 stratified questions; full = all 50 (default: EVAL_MODE).")
    parser.add_argument("--pipeline", choices=PIPELINES, default="baseline",
                        help="Pipeline to evaluate (default: baseline).")
    parser.add_argument("--dataset", type=Path, default=GOLDEN_DATASET_PATH,
                        help="Path to the golden dataset JSON.")
    parser.add_argument("--from-file", type=Path, default=None,
                        help="Evaluate pre-generated outputs instead of running a pipeline.")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH,
                        help="Results JSON to merge into.")
    parser.add_argument("--no-save", action="store_true", help="Do not write results to disk.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate dataset and show the plan without any API calls.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from amorphnet.config import get_settings

    mode = args.mode or get_settings().EVAL_MODE
    full_items = load_golden_dataset(args.dataset)
    problems = validate_dataset(full_items)
    for problem in problems:
        logger.warning("Dataset check: %s", problem)

    items = get_eval_subset(mode, args.dataset)
    print(f"Golden dataset: {len(full_items)} items {distribution(full_items)}")
    print(f"Mode '{mode}': {len(items)} questions {distribution(items)}; "
          f"~{len(items) * LLM_CALLS_PER_QUESTION} judge LLM calls "
          f"(12 RPM => ~{len(items) * LLM_CALLS_PER_QUESTION / 12:.0f}+ min)")

    if args.dry_run:
        print("Dry run: no API calls made.")
        return 1 if problems else 0

    if args.from_file:
        samples = _from_file_samples(args.from_file, items)
    else:
        pipeline_fn = {
            "oracle": lambda: _oracle_pipeline(items),
            "baseline": _baseline_pipeline,
            "full": _full_pipeline,
        }[args.pipeline]()
        samples = asyncio.run(
            generate_samples(
                pipeline_fn,
                items,
                on_progress=lambda i, n, it: print(f"  [{i}/{n}] answered ({it.query_type})"),
            )
        )

    report = evaluate_samples(samples, pipeline=args.pipeline, mode=mode)
    print_summary(report)
    if not args.no_save:
        persist_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
