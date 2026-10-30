"""Every number quoted in the README and the docs is recomputed here."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

from llm_finetune_evaluator import stats
from llm_finetune_evaluator.analysis import AnalysisSettings, analyze
from llm_finetune_evaluator.config import load_config
from llm_finetune_evaluator.report import formatting as fmt
from llm_finetune_evaluator.runner import run_experiment
from llm_finetune_evaluator.storage import load_run

ROOT = Path(__file__).resolve().parents[1]


def _text(path: str) -> str:
    """File contents with whitespace collapsed, so line breaks do not matter."""
    return " ".join((ROOT / path).read_text(encoding="utf-8").split())


def _worked_example() -> tuple[np.ndarray, np.ndarray]:
    both, base_only, tuned_only, neither = 138, 2, 12, 48
    base = np.array([1] * both + [1] * base_only + [0] * tuned_only + [0] * neither, dtype=float)
    tuned = np.array([1] * both + [0] * base_only + [1] * tuned_only + [0] * neither, dtype=float)
    return base, tuned


def test_worked_example() -> None:
    base, tuned = _worked_example()
    assert (base.mean(), tuned.mean()) == (0.70, 0.75)
    pooled = (base.sum() + tuned.sum()) / 400
    z = (tuned.mean() - base.mean()) / math.sqrt(pooled * (1 - pooled) * 2 / 200)
    p_unpaired = 2 * (1 - NormalDist().cdf(z))
    assert round(z, 2) == 1.12
    assert round(p_unpaired, 2) == 0.26
    mcnemar = stats.mcnemar_exact(tuned, base)
    assert (mcnemar.a_only, mcnemar.b_only, round(mcnemar.p_value, 3)) == (12, 2, 0.013)
    assert stats.paired_permutation_test(tuned, base).p_value == pytest.approx(mcnemar.p_value)
    paired = stats.paired_bootstrap_ci(tuned, base, seed=0)
    assert (round(100 * paired.low, 1), round(100 * paired.high, 1)) == (1.5, 8.5)
    base_means = stats._bootstrap_means(base, 10_000, np.random.default_rng(1))
    tuned_means = stats._bootstrap_means(tuned, 10_000, np.random.default_rng(2))
    low, high = np.percentile(tuned_means - base_means, [2.5, 97.5])
    assert (round(100 * low, 1), round(100 * high, 1)) == (-3.5, 13.5)

    readme = _text("README.md")
    for quoted in ("p = 0.26", "p = 0.013", "+1.5 to +8.5 points", "agree on 186 of them"):
        assert quoted in readme
    doc = _text("docs/paired-tests.md")
    for quoted in (
        "z = 1.12 and p = 0.26",
        "\u22123.5 to +13.5 points",
        "p = 0.013",
        "+1.5 to +8.5 points",
    ):
        assert quoted in doc


def test_sizing_tables() -> None:
    doc = _text("docs/sizing-eval-sets.md")
    z = NormalDist().inv_cdf(0.975)
    z_power = NormalDist().inv_cdf(0.8)
    for n in (100, 200, 300, 500, 1000, 2000, 5000):
        cells = []
        for discordance in (0.05, 0.10, 0.20, 0.30):
            mde = stats.minimum_detectable_effect(n, discordance)
            cells.append("not detectable" if mde is None else fmt.mde(mde))
        assert f"| {n:,} | " + " | ".join(cells) + " |" in doc
        half_width = z * math.sqrt(0.8 * 0.2 / n)
        assert f"| {n:,} | {fmt.mde(half_width)} |" in doc
    unpaired_300 = (z + z_power) * math.sqrt(2 * 0.75 * 0.25 / 300)
    unpaired_1000 = (z + z_power) * math.sqrt(2 * 0.75 * 0.25 / 1000)
    assert f"\u00b1{100 * unpaired_300:.1f} points at 300 items" in doc
    assert f"\u00b1{100 * unpaired_1000:.1f} at 1,000 items" in doc
    for effect, column in ((0.02, "2 points"), (0.03, "3 points"), (0.05, "5 points")):
        row = f"| {column} | {stats.required_sample_size(effect, 0.10):,} items | "
        row += f"{stats.required_sample_size(effect, 0.20):,} items |"
        assert row in doc
    bounds = (abs(0.90 - 0.92), 0.90 * (1 - 0.92) + 0.92 * (1 - 0.90))
    assert (round(100 * bounds[0]), round(100 * bounds[1], 1)) == (2, 16.4)


def test_case_study_sizing_claims() -> None:
    doc = _text("docs/sizing-eval-sets.md")
    discordance = 90 / 3080
    assert round(discordance, 4) == 0.0292
    full = stats.minimum_detectable_effect(3080, 0.0292)
    assert full is not None
    assert fmt.mde(full) == "\u00b10.9 pts"
    needed = stats.required_sample_size(0.0065, 0.0292)
    assert round(needed, -2) == 5400
    sliced = stats.minimum_detectable_effect(300, 10 / 300)
    assert sliced is not None
    assert fmt.mde(sliced) == "\u00b12.9 pts"
    assert "MDE of that test set was \u00b10.9 points" in doc
    assert "about 5,400 messages" in doc
    assert "MDE of \u00b12.9 points" in doc
    banking = _text("examples/banking77/README.md")
    assert "about 5,400 messages for 80% power" in banking


def test_kappa_example() -> None:
    human = ["correct"] * 80 + ["incorrect"] * 20
    judge = ["correct"] * 76 + ["incorrect"] * 4 + ["correct"] * 14 + ["incorrect"] * 6
    agreement = sum(h == j for h, j in zip(human, judge, strict=True)) / 100
    assert agreement == 0.82
    assert round(stats.cohen_kappa(human, judge), 2) == 0.31
    assert stats.cohen_kappa(human, ["correct"] * 100) == 0.0
    doc = _text("docs/llm-judges.md")
    assert "Agreement is 82%" in doc
    assert "Cohen's kappa is 0.31" in doc


def test_quickstart_claims(tmp_path: Path) -> None:
    config = load_config(ROOT / "examples" / "mock_quickstart" / "config.yaml")
    result = run_experiment(config, output_dir=tmp_path)
    runs = [load_run(result.runs[name]) for name in ("base", "finetuned")]
    analysis = analyze(runs, settings=AnalysisSettings(), calibration=result.calibration)

    def estimate(metric: str, model: str) -> str:
        value = analysis.headline[metric][model].estimate
        assert value is not None
        return fmt.percent(value.value)

    answer = next(c for c in analysis.pairwise if c.metric == "answer_match")
    assert (estimate("answer_match", "base"), estimate("answer_match", "finetuned")) == (
        "55.0%",
        "68.0%",
    )
    assert (fmt.points(answer.difference), fmt.p_value(answer.p_adjusted)) == ("+13.0 pts", "0.004")
    assert (estimate("trap_accuracy", "base"), estimate("trap_accuracy", "finetuned")) == (
        "21.4%",
        "77.1%",
    )
    assert (estimate("needless_refusal", "base"), estimate("needless_refusal", "finetuned")) == (
        "1.3%",
        "7.6%",
    )
    regressions = {(r.metric, r.slice_label): r for r in analysis.regressions}
    assert ("needless_refusal", "all items") in regressions
    loans = regressions[("answer_match", "topic=loans")]
    assert (fmt.points(loans.difference), loans.n) == ("-30.0 pts", 40)
    assert result.calibration is not None
    assert fmt.kappa(result.calibration.kappa) == "0.90"
    assert analysis.judge_status == "reliable"
    assert len(runs[0].records) == 450

    readme = _text("README.md")
    for quoted in (
        "answer match +13.0 points (55.0% to 68.0%, Holm p = 0.004)",
        "trap accuracy 21.4% to 77.1%",
        "needless refusals up from 1.3% to 7.6%",
        "answer match \u221230.0 points",
        "on 40 items",
        "Cohen's kappa 0.90",
        "450 synthetic support items",
    ):
        assert quoted in readme
    judges = _text("docs/llm-judges.md")
    interval = (
        f"{fmt.kappa(result.calibration.kappa_low)} to {fmt.kappa(result.calibration.kappa_high)}"
    )
    assert f"80 labels give kappa 0.90 with a 95% interval of {interval}" in judges


def test_banking77_claims_in_the_readme() -> None:
    readme = _text("README.md")
    for quoted in (
        "LoRA 92.7%, QLoRA 92.7%, full fine-tuning 86.0% and the base model 37.7%",
        "\u22126.7 points, 4 wins against 24 losses, Holm p = 5.4e-04",
        "\u00b12.9 points",
        "LoRA reached 92.1%, QLoRA 91.5%, full fine-tuning 85.1% and the base model 36.3%",
    ):
        assert quoted in readme
