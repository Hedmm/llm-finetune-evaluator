# How to size an eval set

The question to ask before building an eval set is not "how many items can we
label?" but "what is the smallest difference we need to detect?". A model
release that is two points better is worth knowing about; an eval set that can
only see ten-point differences will call it a tie.

All numbers on this page come from `lfe`'s own functions (alpha = 0.05,
two-sided, 80% power) and are recomputed by `tests/test_docs.py`. Reproduce any
of them with `lfe mde`.

## One model: how precise is a score?

The 95% confidence interval of an accuracy of 80% is roughly ±1.96·√(p(1−p)/n):

| Items | CI half-width at 80% accuracy |
|---|---|
| 100 | ±7.8 pts |
| 200 | ±5.5 pts |
| 300 | ±4.5 pts |
| 500 | ±3.5 pts |
| 1,000 | ±2.5 pts |
| 2,000 | ±1.8 pts |
| 5,000 | ±1.1 pts |

Comparing two overlapping intervals is not a test, though. Two models are
compared item by item, and the precision of that comparison depends on how
often they disagree.

## Two models: the minimum detectable effect

For paired 0/1 outcomes the power of McNemar's test depends on the number of
items n and the **discordance** ψ, the share of items on which the models
disagree (one right, one wrong). The minimum detectable effect solves Connor's
(1987) sample-size formula for the difference d:

    n = (z_alpha/2 · √ψ + z_beta · √(ψ − d²))² / d²

| Items | 5% disagree | 10% disagree | 20% disagree | 30% disagree |
|---|---|---|---|---|
| 100 | not detectable | ±8.8 pts | ±12.4 pts | ±15.2 pts |
| 200 | ±4.4 pts | ±6.2 pts | ±8.8 pts | ±10.8 pts |
| 300 | ±3.6 pts | ±5.1 pts | ±7.2 pts | ±8.8 pts |
| 500 | ±2.8 pts | ±4.0 pts | ±5.6 pts | ±6.8 pts |
| 1,000 | ±2.0 pts | ±2.8 pts | ±4.0 pts | ±4.8 pts |
| 2,000 | ±1.4 pts | ±2.0 pts | ±2.8 pts | ±3.4 pts |
| 5,000 | ±0.9 pts | ±1.3 pts | ±1.8 pts | ±2.2 pts |

"Not detectable" means that even a model winning every discordant item would
not reach significance with 80% probability.

For contrast, if the two models were evaluated on different items (an
unpaired comparison of 75% against 75% + d), the minimum detectable effects
would be ±9.9 points at 300 items and ±5.4 at 1,000 items: roughly twice as
large as the paired values at 10% disagreement.

## Items needed for a given difference

`lfe mde --effect 0.03 --discordance 0.10` gives the inverse:

| Difference to detect | 10% disagree | 20% disagree |
|---|---|---|
| 2 points | 1,960 items | 3,923 items |
| 3 points | 870 items | 1,742 items |
| 5 points | 312 items | 626 items |

## Estimating the discordance

- **From a pilot run.** Run both models on 100 to 200 items. The report and
  `lfe compare` show wins and losses for every pair; their sum divided by n is
  the discordance. Checkpoints of the same model typically disagree on few
  items; a fine-tune and its base model on many.
- **Without data,** the discordance lies between |p₁ − p₂| (one model's errors
  are a subset of the other's) and p₁(1 − p₂) + p₂(1 − p₁) (errors independent).
  For two models at 90% and 92% accuracy that is between 2% and 16.4%.

For continuous metrics (judge scores, F1) the analogue uses the standard
deviation of the per-item differences: `lfe mde --n 300 --sd-difference 0.35`.

## Slices need more items than the headline

A slice of 50 items has the MDE of a 50-item eval set, and the Holm correction
across k slices asks the smallest p-value to clear roughly alpha / k. Two
consequences:

- Plan the eval set so that the slices you care about each have enough items,
  not just the total.
- Expect "possible regressions" (significant before correction only) on small
  slices. They are leads for a targeted follow-up eval, not conclusions.

## An example from the case study

In the [LoRA vs QLoRA vs full fine-tuning case study](https://github.com/Hedmm/ml-case-studies/tree/main/lora-vs-qlora-vs-full-finetuning),
LoRA beat QLoRA by 0.65 points on 3,080 test messages (55 against 35
discordant messages, p = 0.045). The discordance was 90 / 3,080 = 2.9%, so the
MDE of that test set was ±0.9 points (`lfe mde --n 3080 --discordance 0.0292`):
the observed gap was below what the test set could reliably detect, and
detecting a gap of that size with 80% power takes about 5,400 messages
(`lfe mde --effect 0.0065 --discordance 0.0292`). On the 300-message slice in
`examples/banking77`, the same comparison has an MDE of ±2.9 points and shows
no difference at all.

## Rules of thumb

- Decide the smallest difference that would change a decision, then size for it.
- Keep the eval set fixed across checkpoints; pairing only works on the same items.
- Report the MDE next to every "no significant difference".
- Add items where it matters (the slices you will be asked about) rather than
  everywhere.
