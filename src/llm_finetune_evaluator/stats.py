"""Statistics for paired model comparisons.

Every model in an experiment answers the same items, so comparisons are paired:
the question is not "is 74% different from 71%?" but "on the items where the two
models disagree, does one of them win more often than chance would allow?".

This module provides the building blocks:

* :func:`bootstrap_ci`: seeded percentile bootstrap interval for a mean.
* :func:`paired_bootstrap_ci`: bootstrap interval for a difference of means,
  resampling items so that the pairing is preserved.
* :func:`mcnemar_exact`: exact McNemar test for paired binary outcomes.
* :func:`paired_permutation_test`: sign-flip permutation test for paired
  continuous scores (exact when feasible, Monte Carlo otherwise).
* :func:`holm_bonferroni`: Holm's step-down correction for multiple comparisons.
* :func:`minimum_detectable_effect` and :func:`required_sample_size`: power
  calculations for sizing an eval set.
* :func:`cohen_kappa`: chance-corrected agreement between two raters.

Every resampling function takes a seed or a :class:`numpy.random.Generator`, so
each interval and p-value in a report can be reproduced exactly. The test suite
checks these functions against SciPy and against published values.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
SeedLike = int | np.random.Generator | None

# Upper bound on the number of cells in one block of the bootstrap resampling
# matrix. Resamples are drawn in blocks so that memory stays bounded for large
# eval sets (10,000 resamples of 3,000 items would otherwise need 240 MB).
_MAX_BLOCK_CELLS = 2_000_000

# Exact integer arithmetic is used for the binomial tail up to this many
# discordant pairs; above it the tail is summed in log space.
_EXACT_BINOMIAL_LIMIT = 20_000

# Work limit (steps x support size) for the exact sign-flip distribution.
_EXACT_PERMUTATION_WORK = 50_000_000

# Relative tolerance used when comparing resampled statistics with the observed
# one, so that floating-point noise does not change a p-value.
_TIE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ConfidenceInterval:
    """A point estimate with a two-sided confidence interval.

    Attributes:
        estimate: The statistic computed on the original sample.
        low: Lower bound of the interval.
        high: Upper bound of the interval.
        confidence: Nominal coverage, for example 0.95.
    """

    estimate: float
    low: float
    high: float
    confidence: float

    def contains(self, value: float) -> bool:
        """Return whether ``value`` lies inside the closed interval."""
        return self.low <= value <= self.high


@dataclass(frozen=True)
class McNemarResult:
    """Result of an exact McNemar test.

    Attributes:
        a_only: Items where the first system succeeded and the second failed.
        b_only: Items where the second system succeeded and the first failed.
        p_value: Two-sided exact p-value.
    """

    a_only: int
    b_only: int
    p_value: float

    @property
    def n_discordant(self) -> int:
        """Number of items on which the two systems disagree."""
        return self.a_only + self.b_only


@dataclass(frozen=True)
class PermutationResult:
    """Result of a paired permutation test.

    Attributes:
        statistic: Observed mean difference ``mean(a - b)``.
        p_value: Two-sided p-value.
        method: ``"exact"`` when the full permutation distribution was computed,
            ``"monte_carlo"`` when it was sampled.
        n_resamples: Number of random sign flips for the Monte Carlo method,
            ``None`` for the exact method.
    """

    statistic: float
    p_value: float
    method: str
    n_resamples: int | None


def as_generator(seed: SeedLike) -> np.random.Generator:
    """Return a NumPy generator for ``seed``.

    Args:
        seed: An integer seed, an existing generator (returned unchanged) or
            ``None`` for fresh OS entropy.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def derive_generator(seed: int, *key: Hashable) -> np.random.Generator:
    """Return a generator that depends only on ``seed`` and ``key``.

    Reports compute many intervals. Deriving each generator from a stable key,
    instead of drawing every interval from one shared stream, means that adding
    a model or a slice does not change any other number in the report.

    Args:
        seed: Experiment-level seed.
        *key: Values that identify the computation, such as a metric name and
            a slice. They are converted with ``repr``.
    """
    digest = _stable_digest(repr(key))
    return np.random.default_rng([seed, digest])


def _stable_digest(text: str) -> int:
    """Return a 64-bit FNV-1a hash of ``text``, stable across processes."""
    value = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 0x100000001B3) % (1 << 64)
    return value


def _as_vector(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _as_paired_vectors(a: ArrayLike, b: ArrayLike) -> tuple[FloatArray, FloatArray]:
    first = _as_vector(a, "a")
    second = _as_vector(b, "b")
    if first.shape != second.shape:
        raise ValueError(
            f"paired samples must have the same length, got {first.size} and {second.size}"
        )
    return first, second


def _check_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")


def _check_resamples(n_resamples: int) -> None:
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be at least 1, got {n_resamples}")


def _bootstrap_means(values: FloatArray, n_resamples: int, rng: np.random.Generator) -> FloatArray:
    """Return the means of ``n_resamples`` bootstrap resamples of ``values``."""
    n = values.size
    means = np.empty(n_resamples, dtype=np.float64)
    block = max(1, _MAX_BLOCK_CELLS // n)
    for start in range(0, n_resamples, block):
        stop = min(n_resamples, start + block)
        indices = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = values[indices].mean(axis=1)
    return means


def _percentile_interval(
    estimate: float, replicates: FloatArray, confidence: float
) -> ConfidenceInterval:
    tail = (1.0 - confidence) / 2.0
    low, high = np.percentile(replicates, [100.0 * tail, 100.0 * (1.0 - tail)])
    return ConfidenceInterval(
        estimate=estimate, low=float(low), high=float(high), confidence=confidence
    )


def bootstrap_ci(
    values: ArrayLike,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: SeedLike = 0,
) -> ConfidenceInterval:
    """Percentile bootstrap confidence interval for the mean of ``values``.

    Items are resampled with replacement ``n_resamples`` times; the interval is
    given by the empirical quantiles of the resampled means.

    Args:
        values: Per-item scores, for example 0/1 correctness.
        confidence: Nominal two-sided coverage.
        n_resamples: Number of bootstrap resamples.
        seed: Seed or generator for the resampling.

    Returns:
        The sample mean with its percentile interval.
    """
    _check_confidence(confidence)
    _check_resamples(n_resamples)
    sample = _as_vector(values, "values")
    estimate = float(sample.mean())
    if sample.size == 1 or np.all(sample == sample[0]):
        return ConfidenceInterval(estimate, estimate, estimate, confidence)
    replicates = _bootstrap_means(sample, n_resamples, as_generator(seed))
    return _percentile_interval(estimate, replicates, confidence)


def paired_bootstrap_ci(
    a: ArrayLike,
    b: ArrayLike,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: SeedLike = 0,
) -> ConfidenceInterval:
    """Paired bootstrap confidence interval for ``mean(a) - mean(b)``.

    Each resample draws items, not scores, so both systems are always evaluated
    on the same resampled items. Because the mean is linear, this is the same as
    bootstrapping the per-item differences ``a - b``.

    Args:
        a: Per-item scores of the first system.
        b: Per-item scores of the second system, aligned with ``a``.
        confidence: Nominal two-sided coverage.
        n_resamples: Number of bootstrap resamples.
        seed: Seed or generator for the resampling.

    Returns:
        The observed difference of means with its percentile interval.
    """
    first, second = _as_paired_vectors(a, b)
    return bootstrap_ci(first - second, confidence=confidence, n_resamples=n_resamples, seed=seed)


def bootstrap_ci_statistic(
    n_items: int,
    statistic: Callable[[IntArray], float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    seed: SeedLike = 0,
) -> ConfidenceInterval:
    """Percentile bootstrap interval for an arbitrary statistic of items.

    Use this for statistics that are not a mean, such as Cohen's kappa. The
    callable receives an array of item indices (with repetitions) and returns
    the statistic computed on those items. Resamples where the statistic is not
    finite are dropped.

    Args:
        n_items: Number of items in the original sample.
        statistic: Function from an index array to a float.
        confidence: Nominal two-sided coverage.
        n_resamples: Number of bootstrap resamples.
        seed: Seed or generator for the resampling.

    Returns:
        The statistic on the full sample with its percentile interval. The
        bounds are NaN when no resample produced a finite value.
    """
    _check_confidence(confidence)
    _check_resamples(n_resamples)
    if n_items <= 0:
        raise ValueError("n_items must be positive")
    rng = as_generator(seed)
    estimate = float(statistic(np.arange(n_items, dtype=np.int64)))
    replicates = np.array(
        [statistic(rng.integers(0, n_items, size=n_items)) for _ in range(n_resamples)],
        dtype=np.float64,
    )
    finite = replicates[np.isfinite(replicates)]
    if finite.size == 0:
        return ConfidenceInterval(estimate, math.nan, math.nan, confidence)
    return _percentile_interval(estimate, finite, confidence)


def _binary_vector(values: ArrayLike, name: str) -> NDArray[np.bool_]:
    array = _as_vector(values, name)
    if not np.all((array == 0.0) | (array == 1.0)):
        raise ValueError(f"{name} must contain only 0 and 1")
    return np.asarray(array == 1.0, dtype=np.bool_)


def binomial_tail_half(k: int, n: int) -> float:
    """Return ``P(X <= k)`` for ``X ~ Binomial(n, 1/2)``.

    Up to 20,000 trials the sum is computed with exact integer arithmetic and a
    single correctly rounded division, so even p-values around 1e-40 are
    accurate. Larger ``n`` is summed in log space.

    Args:
        k: Upper end of the tail, inclusive.
        n: Number of trials.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if n <= _EXACT_BINOMIAL_LIMIT:
        total = 0
        coefficient = 1
        for i in range(k + 1):
            total += coefficient
            coefficient = coefficient * (n - i) // (i + 1)
        return total / (1 << n)
    log_terms = np.array(
        [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in range(k + 1)]
    )
    peak = float(log_terms.max())
    log_sum = peak + math.log(float(np.exp(log_terms - peak).sum()))
    return min(1.0, math.exp(log_sum - n * math.log(2.0)))


def mcnemar_exact(a: ArrayLike, b: ArrayLike) -> McNemarResult:
    """Exact McNemar test for two systems scored 0/1 on the same items.

    Only discordant items carry information. Under the null hypothesis that
    both systems are equally good, each discordant item is equally likely to
    favour either system, so the count ``a_only`` follows
    ``Binomial(a_only + b_only, 1/2)``. The two-sided p-value doubles the
    smaller tail; this matches ``scipy.stats.binomtest(a_only, n, 0.5)``.

    Args:
        a: 0/1 outcomes of the first system.
        b: 0/1 outcomes of the second system, aligned with ``a``.

    Returns:
        Discordant counts and the two-sided exact p-value.
    """
    first = _binary_vector(a, "a")
    second = _binary_vector(b, "b")
    if first.shape != second.shape:
        raise ValueError(
            f"paired samples must have the same length, got {first.size} and {second.size}"
        )
    a_only = int(np.sum(first & ~second))
    b_only = int(np.sum(~first & second))
    n_discordant = a_only + b_only
    if n_discordant == 0:
        return McNemarResult(a_only, b_only, 1.0)
    p_value = min(1.0, 2.0 * binomial_tail_half(min(a_only, b_only), n_discordant))
    return McNemarResult(a_only, b_only, p_value)


def _integer_grid(values: FloatArray) -> IntArray | None:
    """Express positive ``values`` as integer multiples of a common unit.

    The unit is the smallest value divided by a small integer, so that grids
    such as {0.5, 1} or {0.2, 0.3} are recognised. Returns ``None`` when the
    values do not lie on such a grid, for example for arbitrary F1 differences.
    """
    smallest = float(np.min(values))
    for divisor in range(1, 13):
        scaled = values * (divisor / smallest)
        rounded = np.rint(scaled)
        if np.max(np.abs(scaled - rounded)) <= 1e-6:
            return None if rounded.max() > 10_000 else rounded.astype(np.int64)
    return None


def _exact_sign_flip_pvalue(magnitudes: IntArray, observed: int) -> float | None:
    """Exact two-sided p-value of a sign-flip test on integer magnitudes.

    Under the null hypothesis each magnitude enters the sum with a random sign.
    The distribution of the signed sum is built by dynamic programming over its
    support; probabilities are halved at every step, so nothing overflows.

    Returns ``None`` when the computation would exceed the work limit.
    """
    total = int(magnitudes.sum())
    width = 2 * total + 1
    if magnitudes.size * width > _EXACT_PERMUTATION_WORK:
        return None
    distribution: FloatArray = np.zeros(width, dtype=np.float64)
    distribution[total] = 1.0
    for magnitude in magnitudes.tolist():
        shifted = np.zeros_like(distribution)
        shifted[magnitude:] += distribution[:-magnitude]
        shifted[:-magnitude] += distribution[magnitude:]
        distribution = 0.5 * shifted
    sums = np.arange(-total, total + 1)
    return float(min(1.0, distribution[np.abs(sums) >= abs(observed)].sum()))


def paired_permutation_test(
    a: ArrayLike,
    b: ArrayLike,
    *,
    n_resamples: int = 10_000,
    seed: SeedLike = 0,
) -> PermutationResult:
    """Paired sign-flip permutation test for ``mean(a) == mean(b)``.

    Under the null hypothesis the two scores of an item are exchangeable, so
    each per-item difference ``a_i - b_i`` is equally likely to have either
    sign. The p-value is the share of sign assignments whose mean difference is
    at least as extreme as the observed one.

    The distribution is computed exactly when the differences lie on a common
    grid (rubric scores such as 0, 0.5 and 1 always do), which also makes the
    test identical to :func:`mcnemar_exact` for 0/1 scores. Otherwise
    ``n_resamples`` random sign flips are drawn and the p-value uses the
    ``(count + 1) / (n_resamples + 1)`` estimator, which is never zero.

    Args:
        a: Per-item scores of the first system.
        b: Per-item scores of the second system, aligned with ``a``.
        n_resamples: Number of random sign flips for the Monte Carlo method.
        seed: Seed or generator for the Monte Carlo method.

    Returns:
        The observed mean difference and its two-sided p-value.
    """
    _check_resamples(n_resamples)
    first, second = _as_paired_vectors(a, b)
    differences = first - second
    statistic = float(differences.mean())
    nonzero = differences[differences != 0.0]
    if nonzero.size == 0:
        return PermutationResult(statistic, 1.0, "exact", None)

    magnitudes = np.abs(nonzero)
    grid = _integer_grid(magnitudes)
    if grid is not None:
        signed = np.where(nonzero > 0, grid, -grid)
        p_exact = _exact_sign_flip_pvalue(grid, int(signed.sum()))
        if p_exact is not None:
            return PermutationResult(statistic, p_exact, "exact", None)

    rng = as_generator(seed)
    observed = abs(float(nonzero.sum()))
    threshold = observed - _TIE_TOLERANCE * max(1.0, observed)
    exceed = 0
    block = max(1, _MAX_BLOCK_CELLS // nonzero.size)
    for start in range(0, n_resamples, block):
        size = min(block, n_resamples - start)
        signs = rng.integers(0, 2, size=(size, nonzero.size)) * 2 - 1
        exceed += int(np.sum(np.abs(signs @ nonzero) >= threshold))
    p_value = (exceed + 1) / (n_resamples + 1)
    return PermutationResult(statistic, p_value, "monte_carlo", n_resamples)


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm's step-down adjusted p-values.

    Holm's method controls the family-wise error rate (the chance of at least
    one false positive among all comparisons) at the chosen level, like the
    Bonferroni correction, but rejects more often. Rejecting every hypothesis
    whose adjusted p-value is below ``alpha`` is equivalent to Holm's procedure
    at level ``alpha``.

    Args:
        p_values: Raw p-values of one family of comparisons.

    Returns:
        Adjusted p-values in the original order.
    """
    m = len(p_values)
    for p in p_values:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-values must be in [0, 1], got {p}")
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running_max
    return adjusted


def _z_scores(alpha: float, power: float) -> tuple[float, float]:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0.5 <= power < 1.0:
        raise ValueError(f"power must be in [0.5, 1), got {power}")
    normal = NormalDist()
    return normal.inv_cdf(1.0 - alpha / 2.0), normal.inv_cdf(power)


def minimum_detectable_effect(
    n: int, discordance: float, *, alpha: float = 0.05, power: float = 0.8
) -> float | None:
    """Smallest accuracy difference a paired test is likely to detect.

    Uses the normal approximation to McNemar's test from Connor (1987),
    ``n = (z_a * sqrt(psi) + z_b * sqrt(psi - d**2))**2 / d**2``, solved for the
    difference ``d``. Here ``psi`` is the discordance: the share of items on
    which the two models disagree (one right, one wrong).

    Args:
        n: Number of paired items.
        discordance: Expected share of discordant items, in (0, 1].
        alpha: Two-sided significance level.
        power: Desired probability of detecting the effect.

    Returns:
        The minimum detectable absolute difference in accuracy (0.03 means
        three percentage points), or ``None`` when no difference up to the
        discordance itself is detectable with ``n`` items.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < discordance <= 1.0:
        raise ValueError(f"discordance must be in (0, 1], got {discordance}")
    z_a, z_b = _z_scores(alpha, power)
    radicand = n - z_a**2 + z_b**2
    if radicand < 0:
        return None
    effect = (
        math.sqrt(discordance) * (z_a * math.sqrt(n) + z_b * math.sqrt(radicand)) / (n + z_b**2)
    )
    return effect if effect <= discordance else None


def required_sample_size(
    effect: float, discordance: float, *, alpha: float = 0.05, power: float = 0.8
) -> int:
    """Number of paired items needed to detect an accuracy difference.

    This is the inverse of :func:`minimum_detectable_effect` (Connor, 1987).

    Args:
        effect: Absolute accuracy difference to detect, in (0, discordance].
        discordance: Expected share of items on which the models disagree.
        alpha: Two-sided significance level.
        power: Desired probability of detecting the effect.

    Returns:
        The smallest integer number of items with the requested power.
    """
    if not 0.0 < discordance <= 1.0:
        raise ValueError(f"discordance must be in (0, 1], got {discordance}")
    if not 0.0 < effect <= discordance:
        raise ValueError(
            f"effect must be in (0, discordance], got effect={effect}, discordance={discordance}"
        )
    z_a, z_b = _z_scores(alpha, power)
    root = z_a * math.sqrt(discordance) + z_b * math.sqrt(discordance - effect**2)
    return math.ceil((root / effect) ** 2)


def minimum_detectable_effect_continuous(
    n: int, sd_difference: float, *, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Smallest mean difference a paired test of continuous scores detects.

    Uses the normal approximation ``d = (z_a + z_b) * sd / sqrt(n)``, where
    ``sd`` is the standard deviation of the per-item differences.

    Args:
        n: Number of paired items.
        sd_difference: Standard deviation of the per-item score differences.
        alpha: Two-sided significance level.
        power: Desired probability of detecting the effect.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if sd_difference < 0:
        raise ValueError("sd_difference must be non-negative")
    z_a, z_b = _z_scores(alpha, power)
    return (z_a + z_b) * sd_difference / math.sqrt(n)


def confusion_matrix(
    rater_a: Sequence[str], rater_b: Sequence[str], labels: Sequence[str]
) -> IntArray:
    """Count label pairs: rows are ``rater_a`` labels, columns ``rater_b`` labels.

    Args:
        rater_a: Labels from the first rater.
        rater_b: Labels from the second rater, aligned with ``rater_a``.
        labels: Category order for rows and columns. Every label used by either
            rater must be listed.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("raters must label the same number of items")
    position = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for first, second in zip(rater_a, rater_b, strict=True):
        if first not in position or second not in position:
            raise ValueError(f"label pair ({first!r}, {second!r}) is not in {list(labels)}")
        matrix[position[first], position[second]] += 1
    return matrix


def cohen_kappa(
    rater_a: Sequence[str], rater_b: Sequence[str], labels: Sequence[str] | None = None
) -> float:
    """Cohen's kappa: agreement between two raters corrected for chance.

    ``kappa = (p_o - p_e) / (1 - p_e)``, where ``p_o`` is the observed agreement
    and ``p_e`` the agreement expected if both raters labelled at random with
    their own label frequencies. 1 means perfect agreement and 0 means no better
    than chance.

    Args:
        rater_a: Labels from the first rater, for example a human.
        rater_b: Labels from the second rater, for example an LLM judge.
        labels: Optional category list; defaults to every label that occurs.

    Returns:
        Kappa, or NaN when it is undefined because both raters used one and
        the same category for every item.
    """
    if len(rater_a) == 0:
        raise ValueError("at least one labelled item is required")
    categories = list(labels) if labels is not None else sorted(set(rater_a) | set(rater_b))
    matrix = confusion_matrix(rater_a, rater_b, categories).astype(np.float64)
    n = matrix.sum()
    observed = float(np.trace(matrix) / n)
    expected = float((matrix.sum(axis=1) @ matrix.sum(axis=0)) / n**2)
    if math.isclose(expected, 1.0):
        return math.nan
    return (observed - expected) / (1.0 - expected)
