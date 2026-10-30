"""Tests for the statistics module, checked against SciPy and published values."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import numpy as np
import pytest
from scipy import stats as sps
from sklearn.metrics import cohen_kappa_score

from llm_finetune_evaluator import stats
from llm_finetune_evaluator.stats import FloatArray


def _paired_binary(
    a_only: int, b_only: int, both: int = 7, neither: int = 4
) -> tuple[list[int], list[int]]:
    a = [1] * a_only + [0] * b_only + [1] * both + [0] * neither
    b = [0] * a_only + [1] * b_only + [1] * both + [0] * neither
    return a, b


class TestBootstrap:
    def test_is_reproducible_with_the_same_seed(self) -> None:
        values = np.random.default_rng(0).random(250)
        assert stats.bootstrap_ci(values, seed=5) == stats.bootstrap_ci(values, seed=5)

    def test_changes_with_the_seed(self) -> None:
        values = np.random.default_rng(0).random(250)
        assert stats.bootstrap_ci(values, seed=5) != stats.bootstrap_ci(values, seed=6)

    def test_constant_sample_has_zero_width(self) -> None:
        interval = stats.bootstrap_ci([1.0] * 30)
        assert (interval.estimate, interval.low, interval.high) == (1.0, 1.0, 1.0)

    def test_single_value(self) -> None:
        interval = stats.bootstrap_ci([0.25])
        assert (interval.low, interval.high) == (0.25, 0.25)

    def test_matches_scipy_percentile_bootstrap(self) -> None:
        values = (np.random.default_rng(1).random(400) < 0.7).astype(float)
        ours = stats.bootstrap_ci(values, n_resamples=20_000, seed=2)
        reference = sps.bootstrap(
            (values,), np.mean, method="percentile", n_resamples=20_000, random_state=3
        ).confidence_interval
        assert ours.estimate == pytest.approx(values.mean())
        assert ours.low == pytest.approx(reference.low, abs=0.006)
        assert ours.high == pytest.approx(reference.high, abs=0.006)

    def test_is_close_to_the_wald_interval_for_a_large_proportion_sample(self) -> None:
        values = np.array([1.0] * 720 + [0.0] * 280)
        interval = stats.bootstrap_ci(values, n_resamples=20_000, seed=0)
        half_width = NormalDist().inv_cdf(0.975) * math.sqrt(0.72 * 0.28 / 1000)
        assert interval.low == pytest.approx(0.72 - half_width, abs=0.004)
        assert interval.high == pytest.approx(0.72 + half_width, abs=0.004)

    def test_coverage_is_close_to_nominal(self) -> None:
        rng = np.random.default_rng(11)
        covered = 0
        trials = 300
        for trial in range(trials):
            sample = (rng.random(150) < 0.6).astype(float)
            interval = stats.bootstrap_ci(sample, n_resamples=2_000, seed=trial)
            covered += interval.contains(0.6)
        assert 0.90 <= covered / trials <= 0.98

    def test_honours_the_confidence_level(self) -> None:
        values = np.random.default_rng(2).random(300)
        wide = stats.bootstrap_ci(values, confidence=0.99)
        narrow = stats.bootstrap_ci(values, confidence=0.80)
        assert wide.low < narrow.low < narrow.high < wide.high

    def test_processes_large_samples_in_blocks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(stats, "_MAX_BLOCK_CELLS", 1_000)
        values = np.random.default_rng(3).random(500)
        interval = stats.bootstrap_ci(values, n_resamples=3_001, seed=0)
        assert interval.low < values.mean() < interval.high

    @pytest.mark.parametrize(
        ("bad", "message"),
        [([], "empty"), ([1.0, math.nan], "NaN"), ([[1.0, 0.0]], "one-dimensional")],
    )
    def test_rejects_invalid_input(self, bad: Any, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            stats.bootstrap_ci(bad)

    @pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5])
    def test_rejects_invalid_confidence(self, confidence: float) -> None:
        with pytest.raises(ValueError, match="confidence"):
            stats.bootstrap_ci([0.0, 1.0], confidence=confidence)

    def test_rejects_zero_resamples(self) -> None:
        with pytest.raises(ValueError, match="n_resamples"):
            stats.bootstrap_ci([0.0, 1.0], n_resamples=0)


class TestPairedBootstrap:
    def test_estimate_is_the_difference_of_means(self) -> None:
        a = [1, 1, 0, 1, 0, 1]
        b = [0, 1, 0, 0, 0, 1]
        interval = stats.paired_bootstrap_ci(a, b)
        assert interval.estimate == pytest.approx(np.mean(a) - np.mean(b))

    def test_matches_scipy_paired_bootstrap(self) -> None:
        rng = np.random.default_rng(4)
        a = rng.random(300)
        b = np.clip(a + rng.normal(0.05, 0.2, size=300), 0, 1)

        def difference(x: FloatArray, y: FloatArray, axis: int = -1) -> FloatArray:
            return np.asarray(np.mean(x, axis=axis) - np.mean(y, axis=axis))

        reference = sps.bootstrap(
            (a, b),
            difference,
            paired=True,
            vectorized=True,
            method="percentile",
            n_resamples=20_000,
            random_state=1,
        ).confidence_interval
        ours = stats.paired_bootstrap_ci(a, b, n_resamples=20_000, seed=0)
        assert ours.low == pytest.approx(reference.low, abs=0.004)
        assert ours.high == pytest.approx(reference.high, abs=0.004)

    def test_is_narrower_than_an_unpaired_interval_for_correlated_scores(self) -> None:
        rng = np.random.default_rng(5)
        difficulty = rng.random(400)
        a = (difficulty < 0.70).astype(float)
        b = (difficulty < 0.75).astype(float)
        paired = stats.paired_bootstrap_ci(b, a, seed=0)
        # Unpaired: resample the two systems independently.
        resample_a = stats._bootstrap_means(a, 10_000, np.random.default_rng(1))
        resample_b = stats._bootstrap_means(b, 10_000, np.random.default_rng(2))
        unpaired_low, unpaired_high = np.percentile(resample_b - resample_a, [2.5, 97.5])
        assert (paired.high - paired.low) < 0.5 * (unpaired_high - unpaired_low)

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            stats.paired_bootstrap_ci([1, 0], [1])


class TestBootstrapStatistic:
    def test_bootstraps_an_arbitrary_statistic(self) -> None:
        values = np.random.default_rng(6).random(200)
        interval = stats.bootstrap_ci_statistic(
            values.size, lambda idx: float(np.median(values[idx])), n_resamples=1_000
        )
        assert interval.estimate == pytest.approx(np.median(values))
        assert interval.low < interval.estimate < interval.high

    def test_returns_nan_bounds_when_every_resample_is_undefined(self) -> None:
        interval = stats.bootstrap_ci_statistic(5, lambda idx: math.nan, n_resamples=10)
        assert math.isnan(interval.low)
        assert math.isnan(interval.high)


class TestMcNemar:
    @pytest.mark.parametrize(
        ("a_only", "b_only"),
        [(0, 0), (1, 0), (0, 5), (3, 3), (10, 2), (17, 40), (55, 35), (254, 39), (500, 450)],
    )
    def test_matches_scipy_binomtest(self, a_only: int, b_only: int) -> None:
        a, b = _paired_binary(a_only, b_only)
        result = stats.mcnemar_exact(a, b)
        expected = sps.binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
        assert (result.a_only, result.b_only) == (a_only, b_only)
        assert result.p_value == pytest.approx(expected, rel=1e-9)

    def test_reproduces_the_case_study_p_values(self) -> None:
        # Discordant counts from the LoRA vs QLoRA vs full fine-tuning case study
        # (Banking77, 3,080 test messages); its analyze.py printed these p-values.
        published = {(55, 35): 4.46e-02, (254, 39): 8.30e-40, (247, 52): 1.50e-31}
        for (a_only, b_only), p_value in published.items():
            result = stats.mcnemar_exact(*_paired_binary(a_only, b_only))
            assert result.p_value == pytest.approx(p_value, rel=1e-2)

    def test_is_symmetric(self) -> None:
        a, b = _paired_binary(12, 30)
        assert stats.mcnemar_exact(a, b).p_value == stats.mcnemar_exact(b, a).p_value

    def test_rejects_non_binary_scores(self) -> None:
        with pytest.raises(ValueError, match="only 0 and 1"):
            stats.mcnemar_exact([0.5, 1.0], [1.0, 1.0])

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            stats.mcnemar_exact([1, 0, 1], [1, 0])


class TestBinomialTail:
    @pytest.mark.parametrize(("k", "n"), [(0, 1), (3, 10), (50, 100), (200, 1000), (9_990, 20_000)])
    def test_exact_branch_matches_scipy(self, k: int, n: int) -> None:
        assert stats.binomial_tail_half(k, n) == pytest.approx(sps.binom.cdf(k, n, 0.5), rel=1e-9)

    @pytest.mark.parametrize(("k", "n"), [(14_800, 30_000), (24_000, 50_000)])
    def test_log_space_branch_matches_scipy(self, k: int, n: int) -> None:
        assert stats.binomial_tail_half(k, n) == pytest.approx(sps.binom.cdf(k, n, 0.5), rel=1e-8)

    def test_edge_cases(self) -> None:
        assert stats.binomial_tail_half(-1, 5) == 0.0
        assert stats.binomial_tail_half(5, 5) == 1.0
        assert stats.binomial_tail_half(0, 0) == 1.0


class TestPermutationTest:
    def test_exact_path_matches_scipy_exact_enumeration(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.choice([0.0, 0.5, 1.0], size=12)
        b = rng.choice([0.0, 0.5, 1.0], size=12)
        ours = stats.paired_permutation_test(a, b)
        reference = sps.permutation_test(
            (a, b),
            lambda x, y, axis=-1: np.mean(x - y, axis=axis),
            permutation_type="samples",
            n_resamples=np.inf,
            vectorized=True,
        )
        assert ours.method == "exact"
        assert ours.p_value == pytest.approx(reference.pvalue, rel=1e-9)

    def test_detects_grids_with_a_finer_unit_than_the_smallest_difference(self) -> None:
        a = [0.3, 0.2, 0.5, 0.0, 0.3]
        b = [0.0, 0.0, 0.0, 0.0, 0.0]
        result = stats.paired_permutation_test(a, b)
        reference = sps.permutation_test(
            (np.array(a), np.array(b)),
            lambda x, y, axis=-1: np.mean(x - y, axis=axis),
            permutation_type="samples",
            n_resamples=np.inf,
            vectorized=True,
        )
        assert result.method == "exact"
        assert result.p_value == pytest.approx(reference.pvalue, rel=1e-9)

    def test_monte_carlo_path_agrees_with_scipy(self) -> None:
        rng = np.random.default_rng(8)
        a = rng.random(80)
        b = np.clip(a - rng.normal(0.03, 0.15, size=80), 0, 1)
        ours = stats.paired_permutation_test(a, b, n_resamples=40_000, seed=0)
        reference = sps.permutation_test(
            (a, b),
            lambda x, y, axis=-1: np.mean(x - y, axis=axis),
            permutation_type="samples",
            n_resamples=40_000,
            vectorized=True,
            random_state=1,
        )
        assert ours.method == "monte_carlo"
        tolerance = 4 * math.sqrt(reference.pvalue * (1 - reference.pvalue) / 40_000) + 1e-3
        assert ours.p_value == pytest.approx(reference.pvalue, abs=tolerance)

    @pytest.mark.parametrize(("a_only", "b_only"), [(9, 2), (30, 18), (120, 95), (0, 4)])
    def test_equals_exact_mcnemar_on_binary_scores(self, a_only: int, b_only: int) -> None:
        a, b = _paired_binary(a_only, b_only)
        permutation = stats.paired_permutation_test(a, b)
        assert permutation.method == "exact"
        assert permutation.p_value == pytest.approx(stats.mcnemar_exact(a, b).p_value, rel=1e-9)

    def test_identical_systems_have_p_value_one(self) -> None:
        scores = [0.1, 0.5, 0.9]
        result = stats.paired_permutation_test(scores, scores)
        assert (result.statistic, result.p_value) == (0.0, 1.0)

    def test_monte_carlo_p_value_is_never_zero(self) -> None:
        rng = np.random.default_rng(9)
        a = rng.random(200) + 1.0
        b = rng.random(200)
        result = stats.paired_permutation_test(a, b, n_resamples=999, seed=0)
        assert result.p_value == pytest.approx(1 / 1000)

    def test_is_reproducible(self) -> None:
        rng = np.random.default_rng(10)
        a, b = rng.random(50), rng.random(50)
        first = stats.paired_permutation_test(a, b, seed=3)
        assert first == stats.paired_permutation_test(a, b, seed=3)

    def test_statistic_is_the_mean_difference(self) -> None:
        result = stats.paired_permutation_test([1.0, 0.5, 0.0, 1.0], [0.5, 0.5, 0.0, 0.0])
        assert result.statistic == pytest.approx(0.375)


class TestHolmBonferroni:
    def test_textbook_example(self) -> None:
        # Four hypotheses with p = 0.01, 0.04, 0.03 and 0.005: at alpha = 0.05
        # Holm rejects the first and the fourth only.
        adjusted = stats.holm_bonferroni([0.01, 0.04, 0.03, 0.005])
        assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])
        assert [p < 0.05 for p in adjusted] == [True, False, False, True]

    def test_agrees_with_the_step_down_procedure(self) -> None:
        rng = np.random.default_rng(12)
        for _ in range(200):
            p_values = list(rng.random(8) ** 3)
            adjusted = stats.holm_bonferroni(p_values)
            rejected = set()
            for rank, index in enumerate(np.argsort(p_values)):
                if p_values[index] > 0.05 / (len(p_values) - rank):
                    break
                rejected.add(int(index))
            assert rejected == {i for i, p in enumerate(adjusted) if p <= 0.05}

    def test_adjusted_values_are_bounded(self) -> None:
        p_values = [0.2, 0.9, 0.0001, 0.5]
        adjusted = stats.holm_bonferroni(p_values)
        assert all(raw <= adj <= 1.0 for raw, adj in zip(p_values, adjusted, strict=True))

    def test_single_and_empty_families(self) -> None:
        assert stats.holm_bonferroni([0.03]) == [0.03]
        assert stats.holm_bonferroni([]) == []

    def test_rejects_invalid_p_values(self) -> None:
        with pytest.raises(ValueError, match="p-values"):
            stats.holm_bonferroni([0.5, 1.5])


class TestPower:
    def test_required_sample_size_known_value(self) -> None:
        # (1.959964 * sqrt(0.15) + 0.841621 * sqrt(0.1475))**2 / 0.05**2 = 468.6
        assert stats.required_sample_size(0.05, 0.15) == 469

    def test_mde_solves_connors_equation(self) -> None:
        z_a, z_b = NormalDist().inv_cdf(0.975), NormalDist().inv_cdf(0.8)
        for n, psi in [(100, 0.3), (300, 0.12), (3080, 0.095), (50_000, 0.02)]:
            delta = stats.minimum_detectable_effect(n, psi)
            assert delta is not None
            implied_n = (z_a * math.sqrt(psi) + z_b * math.sqrt(psi - delta**2)) ** 2 / delta**2
            assert implied_n == pytest.approx(n, rel=1e-9)

    def test_round_trips_with_required_sample_size(self) -> None:
        delta = stats.minimum_detectable_effect(800, 0.2, alpha=0.01, power=0.9)
        assert delta is not None
        assert stats.required_sample_size(delta, 0.2, alpha=0.01, power=0.9) == 800

    def test_large_n_approaches_the_normal_approximation(self) -> None:
        z_sum = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.8)
        delta = stats.minimum_detectable_effect(1_000_000, 0.1)
        assert delta == pytest.approx(z_sum * math.sqrt(0.1 / 1_000_000), rel=1e-3)

    def test_returns_none_when_nothing_is_detectable(self) -> None:
        assert stats.minimum_detectable_effect(50, 0.01) is None
        assert stats.minimum_detectable_effect(2, 0.5) is None

    def test_simulated_power_of_exact_mcnemar_is_close_to_target(self) -> None:
        n, psi = 400, 0.2
        delta = stats.minimum_detectable_effect(n, psi)
        assert delta is not None
        rng = np.random.default_rng(13)
        trials = 2_000
        rejections = 0
        for _ in range(trials):
            discordant = rng.binomial(n, psi)
            b_only = rng.binomial(discordant, (psi + delta) / (2 * psi))
            a_only = discordant - b_only
            p = min(1.0, 2 * stats.binomial_tail_half(min(a_only, b_only), discordant))
            rejections += p < 0.05
        assert rejections / trials == pytest.approx(0.8, abs=0.06)

    def test_continuous_mde(self) -> None:
        z_sum = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.8)
        assert stats.minimum_detectable_effect_continuous(400, 0.3) == pytest.approx(
            z_sum * 0.3 / 20
        )

    @pytest.mark.parametrize(
        ("effect", "psi", "message"),
        [
            (0.0, 0.1, "effect"),
            (0.2, 0.1, "effect"),
            (0.05, 0.0, "discordance"),
            (0.05, 1.2, "discordance"),
        ],
    )
    def test_required_sample_size_rejects_invalid_inputs(
        self, effect: float, psi: float, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            stats.required_sample_size(effect, psi)

    def test_rejects_invalid_power(self) -> None:
        with pytest.raises(ValueError, match="power"):
            stats.minimum_detectable_effect(100, 0.2, power=0.3)


class TestCohenKappa:
    def test_two_category_textbook_example(self) -> None:
        # 50 items: both "yes" 20, only the first rater "yes" 5, only the second
        # rater "yes" 10, both "no" 15. Observed agreement 0.7, chance 0.5.
        first = ["yes"] * 20 + ["yes"] * 5 + ["no"] * 10 + ["no"] * 15
        second = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
        assert stats.cohen_kappa(first, second) == pytest.approx(0.4)

    def test_matches_scikit_learn(self) -> None:
        rng = np.random.default_rng(14)
        labels = ["correct", "partially_correct", "incorrect", "refused", "made_up"]
        first = list(rng.choice(labels, size=120))
        second = [x if rng.random() < 0.6 else str(rng.choice(labels)) for x in first]
        assert stats.cohen_kappa(first, second) == pytest.approx(cohen_kappa_score(first, second))

    def test_perfect_agreement_is_one(self) -> None:
        labels = ["a", "b", "c", "a"]
        assert stats.cohen_kappa(labels, labels) == pytest.approx(1.0)

    def test_chance_agreement_is_near_zero(self) -> None:
        rng = np.random.default_rng(15)
        first = list(rng.choice(["a", "b", "c"], size=20_000))
        second = list(rng.choice(["a", "b", "c"], size=20_000))
        assert abs(stats.cohen_kappa(first, second)) < 0.03

    def test_is_undefined_when_both_raters_use_one_category(self) -> None:
        assert math.isnan(stats.cohen_kappa(["a"] * 5, ["a"] * 5))

    def test_confusion_matrix_rows_belong_to_the_first_rater(self) -> None:
        matrix = stats.confusion_matrix(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
        assert matrix.tolist() == [[1, 1], [0, 1]]

    def test_confusion_matrix_rejects_unknown_labels(self) -> None:
        with pytest.raises(ValueError, match="not in"):
            stats.confusion_matrix(["a"], ["z"], ["a", "b"])


class TestDerivedGenerators:
    def test_depend_only_on_seed_and_key(self) -> None:
        first = stats.derive_generator(0, "accuracy", "topic=billing").random()
        again = stats.derive_generator(0, "accuracy", "topic=billing").random()
        other = stats.derive_generator(0, "accuracy", "topic=shipping").random()
        reseeded = stats.derive_generator(1, "accuracy", "topic=billing").random()
        assert first == again
        assert len({first, other, reseeded}) == 3
