"""Smoke tests of the ``lfe`` command line, all offline with the mock backend."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from helpers import mock_experiment, read_jsonl, write_jsonl
from llm_finetune_evaluator import __version__
from llm_finetune_evaluator.cli import app

ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = ROOT / "examples" / "mock_quickstart" / "config.yaml"
COMMANDS = ["run", "compare", "report", "judge-calibrate", "mde"]

runner = CliRunner(env={"COLUMNS": "200"})


def invoke(*args: str) -> tuple[int, str]:
    result = runner.invoke(app, list(args))
    return result.exit_code, result.output


def test_help_lists_every_command() -> None:
    code, output = invoke("--help")
    assert code == 0
    for command in COMMANDS:
        assert command in output


@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_help_has_examples(command: str) -> None:
    code, output = invoke(command, "--help")
    assert code == 0
    assert "Examples" in output
    assert f"lfe {command}" in output


def test_version() -> None:
    assert invoke("--version") == (0, f"llm-finetune-evaluator {__version__}\n")


@pytest.fixture(scope="module")
def quickstart_runs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("quickstart") / "runs"
    result = runner.invoke(app, ["run", str(QUICKSTART), "--output-dir", str(out), "--quiet"])
    assert result.exit_code == 0, result.output
    assert "Headline metrics" in result.output
    return out


def test_run_writes_runs_calibration_and_a_summary(quickstart_runs: Path) -> None:
    assert (quickstart_runs / "base" / "run.json").is_file()
    assert (quickstart_runs / "finetuned" / "records.jsonl").is_file()
    assert json.loads((quickstart_runs / "calibration.json").read_text())["reliable"] is True
    assert json.loads((quickstart_runs / "experiment.json").read_text())["baseline"] == "base"


def test_report_writes_markdown_and_html(quickstart_runs: Path) -> None:
    code, output = invoke("report", str(quickstart_runs))
    assert code == 0, output
    html = (quickstart_runs / "report.html").read_text()
    markdown = (quickstart_runs / "report.md").read_text()
    assert "<title>Quickstart: fine-tuned vs base (simulated)</title>" in html
    assert "https://" not in html
    assert markdown.startswith("# Quickstart: fine-tuned vs base (simulated)")
    assert "topic=loans" in markdown
    assert json.loads((quickstart_runs / "analysis.json").read_text())["judge_status"] == "reliable"


def test_report_options(quickstart_runs: Path, tmp_path: Path) -> None:
    code, output = invoke(
        "report",
        str(quickstart_runs),
        "--format",
        "md",
        "--out",
        str(tmp_path),
        "--title",
        "Custom",
    )
    assert code == 0, output
    assert sorted(path.name for path in tmp_path.iterdir()) == ["report.md"]
    assert (tmp_path / "report.md").read_text().startswith("# Custom")
    code, output = invoke("report", str(quickstart_runs), "--format", "pdf")
    assert code == 1
    assert "unknown report format" in output


def test_report_and_compare_can_gate_ci(quickstart_runs: Path, tmp_path: Path) -> None:
    code, _ = invoke("report", str(quickstart_runs), "--out", str(tmp_path), "--fail-on-regression")
    assert code == 2
    code, output = invoke(
        "compare",
        str(quickstart_runs / "base"),
        str(quickstart_runs / "finetuned"),
        "--fail-on-regression",
    )
    assert code == 2
    assert "regression" in output
    assert "topic=loans" in output


def test_compare_writes_reports_on_request(quickstart_runs: Path, tmp_path: Path) -> None:
    code, output = invoke(
        "compare",
        str(quickstart_runs / "base"),
        str(quickstart_runs / "finetuned"),
        "--out",
        str(tmp_path),
    )
    assert code == 0, output
    assert (tmp_path / "report.html").is_file()


def test_compare_runs_with_the_same_model_name(tmp_path: Path) -> None:
    config = mock_experiment(tmp_path, n=40)
    for name in ("first", "second"):
        code, output = invoke("run", str(config), "--output-dir", str(tmp_path / name), "--quiet")
        assert code == 0, output
    code, output = invoke(
        "compare", str(tmp_path / "first" / "base"), str(tmp_path / "second" / "base")
    )
    assert code == 0, output
    assert "second/base" in output


def test_dry_run_calls_no_model(tmp_path: Path) -> None:
    code, output = invoke("run", str(QUICKSTART), "--dry-run")
    assert code == 0, output
    assert "is valid" in output
    assert "First prompt of support_qa" in output
    assert "answer_match" in output
    assert not (Path.cwd() / "runs" / "dry-run-should-not-exist").exists()


def test_run_with_report_flag(tmp_path: Path) -> None:
    config = mock_experiment(tmp_path, n=40)
    code, output = invoke("run", str(config), "--report", "--quiet")
    assert code == 0, output
    assert (tmp_path / "runs" / "report.html").is_file()


def test_judge_calibrate(tmp_path: Path) -> None:
    out = tmp_path / "calibration.json"
    code, output = invoke("judge-calibrate", str(QUICKSTART), "--out", str(out))
    assert code == 0, output
    assert "kappa" in output
    assert json.loads(out.read_text())["n"] == 80


def _noisy_judge_config(tmp_path: Path) -> Path:
    labels = write_jsonl(
        tmp_path / "labels.jsonl",
        [{"id": f"q{i:03d}", "output": f"answer {i}", "label": "correct"} for i in range(20)]
        + [{"id": f"q{i:03d}", "output": "nonsense", "label": "incorrect"} for i in range(20, 40)],
    )
    return mock_experiment(
        tmp_path,
        n=40,
        scorers=[{"type": "judge"}],
        extra={
            "judge": {
                "model": {"backend": "mock_judge", "noise": 0.8},
                "calibration": {"labels": str(labels)},
            }
        },
    )


def test_judge_calibrate_warns_when_the_judge_is_unreliable(tmp_path: Path) -> None:
    config = _noisy_judge_config(tmp_path)
    out = tmp_path / "calibration.json"
    code, output = invoke("judge-calibrate", str(config), "--out", str(out))
    assert code == 0
    assert "WARNING" in output
    assert "too low" in output
    assert json.loads(out.read_text())["reliable"] is False
    code, _ = invoke("judge-calibrate", str(config), "--out", str(out), "--strict")
    assert code == 2


def test_unreliable_judge_is_marked_in_the_report(tmp_path: Path) -> None:
    config = _noisy_judge_config(tmp_path)
    code, output = invoke("run", str(config), "--report", "--quiet")
    assert code == 0, output
    assert "unreliable judge" in output
    assert "unreliable judge" in (tmp_path / "runs" / "report.html").read_text()


def test_export_labelling_template(quickstart_runs: Path, tmp_path: Path) -> None:
    template = tmp_path / "to_label.jsonl"
    code, output = invoke(
        "judge-calibrate",
        str(QUICKSTART),
        "--export-template",
        str(template),
        "--runs-dir",
        str(quickstart_runs),
        "--size",
        "12",
    )
    assert code == 0, output
    rows = read_jsonl(template)
    assert len(rows) == 12
    assert all(row["label"] is None and "model" not in row for row in rows)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--n", "300", "--discordance", "0.15"], "±"),
        (["--effect", "0.03", "--discordance", "0.15"], "paired items"),
        (["--discordance", "0.15"], "5,000"),
        (["--n", "300", "--sd-difference", "0.35"], "\u00b15.7 pts"),
        (["--n", "20", "--discordance", "0.01"], "not detectable"),
    ],
)
def test_mde(args: list[str], expected: str) -> None:
    code, output = invoke("mde", *args)
    assert code == 0, output
    assert expected in output


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["mde", "--n", "300"], "pass --discordance"),
        (["mde", "--effect", "0.5", "--discordance", "0.1"], "effect must be in"),
        (["run", "missing.yaml"], "config file not found"),
        (["report", "does-not-exist"], "does not exist"),
    ],
)
def test_errors_are_reported_without_a_traceback(args: list[str], message: str) -> None:
    code, output = invoke(*args)
    assert code == 1
    assert "error:" in output
    assert message in output
    assert "Traceback" not in output


def test_invalid_config_is_explained(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("name: demo\nmodels: []\n", encoding="utf-8")
    code, output = invoke("run", str(config))
    assert code == 1
    assert "is not a valid config" in output
    assert "eval_sets" in output
