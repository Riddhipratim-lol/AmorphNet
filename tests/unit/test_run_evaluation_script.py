"""Phase 1.3 — ``scripts/run_evaluation.py`` CLI (argument parsing, adapters, persistence)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from amorphnet.evaluation.dataset_builder import get_eval_subset
from amorphnet.evaluation.ragas_runner import RagasReport

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_evaluation.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("run_evaluation_under_test", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _report(pipeline="baseline", mode="smoke"):
    return RagasReport(pipeline=pipeline, mode=mode, num_samples=1, metrics={}, by_query_type={},
                       targets={}, targets_met={}, per_sample=[])


def test_parse_args_defaults(cli):
    a = cli.parse_args([])
    assert a.pipeline == "baseline" and a.mode is None and not a.dry_run and not a.no_save
    assert a.from_file is None and a.output == cli.RESULTS_PATH


@pytest.mark.parametrize("argv", [["--mode", "medium"], ["--pipeline", "magic"]])
def test_parse_args_rejects_invalid_choices(cli, argv):
    with pytest.raises(SystemExit):
        cli.parse_args(argv)


def test_pipelines_exposed(cli):
    assert set(cli.PIPELINES) == {"oracle", "baseline", "full"}


async def test_oracle_pipeline_returns_ground_truth(cli):
    items = get_eval_subset("smoke")
    out = await cli._oracle_pipeline(items)(items[0].question, items[0].filters)
    assert out.answer == items[0].ground_truth and out.contexts == [items[0].ground_truth]


def test_chunk_text_accepts_str_mapping_and_object(cli):
    from types import SimpleNamespace
    assert cli._chunk_text("a") == "a"
    assert cli._chunk_text({"text": "b"}) == "b"
    assert cli._chunk_text(SimpleNamespace(text="c")) == "c"
    assert cli._chunk_text(object()) == ""


def test_baseline_and_full_exit_cleanly_when_not_implemented(cli, monkeypatch):
    import importlib

    def boom(name):
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", boom)
    with pytest.raises(SystemExit):
        cli._baseline_pipeline()
    with pytest.raises(SystemExit):
        cli._full_pipeline()


async def test_baseline_adapter_forwards_to_answer_question(cli, monkeypatch):
    import sys, types
    mod = types.ModuleType("amorphnet.generation.baseline_generator")
    seen = {}

    async def answer_question(q, f):
        seen.update(q=q, f=f)
        return {"answer": "A", "contexts": ["c1", "c2"]}

    mod.answer_question = answer_question
    monkeypatch.setitem(sys.modules, "amorphnet.generation.baseline_generator", mod)
    out = await cli._baseline_pipeline()("Q?", {"fiscal_year": 2025})
    assert (out.answer, out.contexts) == ("A", ["c1", "c2"]) and seen["f"] == {"fiscal_year": 2025}


async def test_full_adapter_maps_langgraph_state(cli, monkeypatch):
    import sys, types
    mod = types.ModuleType("amorphnet.graph.langgraph_pipeline")

    class App:
        async def ainvoke(self, state):
            assert state == {"original_query": "Q?", "filters": {"fiscal_year": 2025}}
            return {"answer": "A", "reranked_chunks": [{"text": "t1"}, "t2"]}

    mod.amorphnet_pipeline = App()
    monkeypatch.setitem(sys.modules, "amorphnet.graph.langgraph_pipeline", mod)
    out = await cli._full_pipeline()("Q?", {"fiscal_year": 2025})
    assert out.answer == "A" and out.contexts == ["t1", "t2"]


def test_from_file_aligns_by_question_then_by_index(cli, tmp_path):
    items = get_eval_subset("smoke")[:2]
    f = tmp_path / "o.json"
    f.write_text(json.dumps([
        {"answer": "positional", "contexts": ["y"]},
        {"question": items[1].question, "answer": "keyed", "contexts": ["x"]},
    ]))
    s = cli._from_file_samples(f, items)
    assert s[1].answer == "keyed" and s[0].answer == "positional"  # keyed hit, positional fallback
    assert s[0].ground_truth == items[0].ground_truth


def test_from_file_errors(cli, tmp_path):
    items = get_eval_subset("smoke")[:2]
    f = tmp_path / "o.json"
    f.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(SystemExit):
        cli._from_file_samples(f, items)
    f.write_text(json.dumps([{"answer": "only one"}]))
    with pytest.raises(SystemExit):
        cli._from_file_samples(f, items)


def test_persist_report_merges_pipeline_and_mode(cli, tmp_path):
    out = tmp_path / "res" / "ablation.json"
    cli.persist_report(_report("baseline", "smoke"), out)
    cli.persist_report(_report("baseline", "full"), out)
    cli.persist_report(_report("full", "smoke"), out)
    data = json.loads(out.read_text())
    assert set(data) == {"baseline", "full"} and set(data["baseline"]) == {"smoke", "full"}


def test_persist_report_handles_empty_existing_file(cli, tmp_path):
    out = tmp_path / "a.json"
    out.write_text("")
    cli.persist_report(_report(), out)
    assert "baseline" in json.loads(out.read_text())


def test_print_summary_renders_all_metrics(cli, capsys):
    r = RagasReport(pipeline="p", mode="smoke", num_samples=2,
                    metrics={"faithfulness": 0.95, "context_recall": None},
                    by_query_type={"semantic_risk": {"faithfulness": 0.9}},
                    targets={"faithfulness": 0.9, "context_recall": 0.8},
                    targets_met={"faithfulness": True, "context_recall": None}, per_sample=[])
    cli.print_summary(r)
    out = capsys.readouterr().out
    assert "PASS" in out and "n/a" in out and "semantic_risk" in out


def test_dry_run_makes_no_api_calls(cli, capsys, monkeypatch):
    monkeypatch.setattr(cli, "evaluate_samples", lambda *a, **k: pytest.fail("must not evaluate"))
    assert cli.main(["--mode", "smoke", "--dry-run"]) == 0
    assert "Dry run" in capsys.readouterr().out


def test_main_full_flow_with_stubbed_judge(cli, tmp_path, monkeypatch):
    captured = {}

    def fake_eval(samples, pipeline, mode):
        captured.update(n=len(samples), pipeline=pipeline, mode=mode)
        return _report(pipeline, mode)

    monkeypatch.setattr(cli, "evaluate_samples", fake_eval)
    out = tmp_path / "r.json"
    assert cli.main(["--pipeline", "oracle", "--mode", "smoke", "--output", str(out)]) == 0
    assert captured == {"n": 10, "pipeline": "oracle", "mode": "smoke"}
    assert out.exists()


def test_no_save_skips_persistence(cli, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "evaluate_samples", lambda s, pipeline, mode: _report(pipeline, mode))
    out = tmp_path / "r.json"
    cli.main(["--pipeline", "oracle", "--mode", "smoke", "--output", str(out), "--no-save"])
    assert not out.exists()
