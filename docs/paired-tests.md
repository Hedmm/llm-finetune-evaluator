# Why paired tests

When two models answer the same eval set, the data come in pairs: for every
item there is the base model's answer and the fine-tune's answer. Treating the
two accuracies as two independent samples throws that pairing away, and with
it most of the statistical power. `lfe` always compares models item by item.

## A worked example

200 questions. Model A answers 140 correctly (70%), model B 150 (75%). Is B
better?

| | B correct | B wrong |
|---|---|---|
| **A correct** | 138 | 2 |
| **A wrong** | 12 | 48 |

**Unpaired view.** Compare 140/200 with 150/200 as if they came from different
question sets: a two-proportion z-test gives z = 1.12 and p = 0.26, and a
bootstrap that resamples each model separately gives a 95% interval of −3.5 to
+13.5 points for the difference. Not significant.

**Paired view.** The models agree on 186 of the 200 questions; those items say
nothing about which model is better. On the 14 questions where they disagree,
B wins 12 and A wins 2. If the models were equally good, each of these 14 would
be a coin flip, and 12 or more heads (or tails) out of 14 happens with
probability 0.013. The exact McNemar test gives p = 0.013, and the paired
bootstrap interval is +1.5 to +8.5 points. Significant.

Same data, same five-point difference, opposite conclusions. The unpaired test
is not wrong, it answers a different question ("would a different set of 200
questions show a gap?") and pays for the variation between question sets that
the paired design removes. The numbers above come from `lfe`'s own functions;
`tests/test_docs.py` recomputes them.

## Why pairing helps so much

The variance of a difference is

    Var(B − A) = Var(B) + Var(A) − 2 Cov(A, B)

Real models agree on which items are easy and which are hard, so their
per-item scores are strongly correlated and the covariance term cancels most of
the variance. The more two models agree, the smaller the difference they can
be told apart by. The mock backend reproduces this correlation with a shared
per-item difficulty, which is why its quickstart behaves like real models.

## The tests `lfe` uses

**Exact McNemar test** for 0/1 metrics (accuracy, trap accuracy, refusals).
Only discordant items count. Under the null hypothesis the number of items won
by one model follows Binomial(n_discordant, 1/2); the two-sided p-value doubles
the smaller tail. `lfe` computes it with exact integer arithmetic, so even
p-values around 1e-40 are accurate (the test suite checks them against SciPy
and against the p-values published in the case study).

**Paired permutation test** for continuous metrics (field-level F1, judge
scores with partial credit). Under the null hypothesis the two scores of an
item are exchangeable, so each per-item difference is equally likely to have
either sign. The p-value is the share of sign flips whose mean difference is at
least as extreme as the observed one. When the differences lie on a grid (judge
scores of 0, 0.5 and 1 always do) the distribution is computed exactly by
dynamic programming; for 0/1 scores this is exactly the McNemar test. Otherwise
10,000 random sign flips are drawn and the p-value uses the
`(count + 1) / (resamples + 1)` estimator, which is never zero.

**Paired bootstrap intervals.** Items, not scores, are resampled, so both
models are always evaluated on the same resampled items. Because the mean is
linear this equals bootstrapping the per-item differences. Every interval is a
seeded percentile bootstrap with 10,000 resamples; each one draws from a
generator derived from the experiment seed and a stable key (metric, models,
slice), so adding a model or a slice never changes any other number.

## Many comparisons: Holm's correction

A report compares several models, metrics and slices. At alpha = 0.05, twenty
independent slice tests would produce one false "regression" on average even
if nothing changed. `lfe` applies Holm's step-down correction, which controls
the family-wise error rate (the chance of any false positive in the family)
like Bonferroni but rejects more often. The families are:

- the pairwise table: all model pairs, separately for each metric;
- the slice tables: all slices of one candidate against the baseline,
  separately for each metric.

Slices that contain exactly the same items (for example `task=trap` and
`difficulty=hard` when every trap question is hard) are tested once and listed
as aliases, so duplicates do not inflate the family.

A **regression** is a comparison against the baseline that is significantly
worse after correction. Slices that are worse before correction only are
listed as **possible regressions**: worth a look, not yet evidence.

## What "not significant" means

A non-significant difference is not evidence that two models are equally good.
Every comparison in the report has a **minimum detectable effect** (MDE): the
smallest difference the test would detect with 80% power at alpha = 0.05,
given the number of items and the observed disagreement rate. If the MDE is
±6 points and you care about 2, the eval set is too small to tell; see
[sizing-eval-sets.md](sizing-eval-sets.md).

## Pitfalls the tool guards against

- **Different items.** Runs are compared only when their eval sets have the same
  content hash; pairing items by id across edited eval sets would be silently
  wrong. Re-running is cheap because generations are cached.
- **Sampling noise.** Greedy decoding (`temperature: 0`) is the default, so a
  difference reflects the models, not one lucky sample.
- **Unscorable items.** An item that one model cannot be scored on (an
  unparseable judge reply) is dropped from that metric for both models.
- **Direction.** Some metrics should go down (needless refusals); every
  comparison knows its metric's direction, so "worse" always means worse.
