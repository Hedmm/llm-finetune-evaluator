# Quickstart: fine-tuned vs base (simulated)

Synthetic customer-support eval for a fictional bank: product-sheet questions with trap questions, intent routing and transfer-request extraction. Simulated models and judge.

Generated 2026-10-30 15:29 UTC by llm-finetune-evaluator 0.1.0. Baseline: **base**. 2 models, 450 items in 3 eval set(s).

## Headline metrics

Mean score per model with 95% percentile bootstrap intervals (10,000 resamples, seed 0).

| Metric | n | base | finetuned |
| --- | ---: | ---: | ---: |
| answer_match | 200 | 55.0% (48.0 to 62.0) | 68.0% (61.5 to 74.5) |
| judge_score | 270 | 46.5% (40.9 to 52.2) | 65.9% (60.6 to 71.1) |
| trap_accuracy | 70 | 21.4% (12.9 to 31.4) | 77.1% (67.1 to 87.1) |
| needless_refusal (lower is better) | 380 | 1.3% (0.3 to 2.6) | 7.6% (5.0 to 10.5) |
| numeric_match | 146 | 56.8% (49.3 to 65.1) | 67.8% (60.3 to 75.3) |
| intent_accuracy | 120 | 60.8% (52.5 to 70.0) | 71.7% (63.3 to 79.2) |
| field_f1 | 60 | 63.3% (54.3 to 72.8) | 89.9% (82.9 to 96.0) |

Metric definitions:

- `answer_match`: Normalised match: the output contains a reference answer.
- `judge_score`: Judge score from mock-judge (correct = 1, partially correct = 0.5, otherwise 0).
- `trap_accuracy`: Trap accuracy: the model abstains when the answer is not in the source..
- `needless_refusal`: Needless refusals: answerable items the model refused or claimed it could not answer..
- `numeric_match`: Numeric match (exact, last number in the output).
- `intent_accuracy`: Label-set accuracy over 8 labels (off-list answers count as wrong).
- `field_f1`: Field F1: field-level F1 of the JSON object in the output against the reference..

## Comparison with the baseline

Paired tests on the items both models answered: exact McNemar for 0/1 scores, a paired permutation test otherwise. p-values are Holm-adjusted across all model pairs of each metric. MDE is the smallest difference this comparison would detect with 80% power at alpha = 0.05.

| Metric | Comparison | Slice | n | Difference | CI | p | p (Holm) | Wins / losses | MDE | Verdict |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| answer_match | finetuned vs base | all items | 200 | +13.0 pts | +4.5 to +21.5 | 0.004 | 0.004 | 52 / 26 | ±12.3 pts | ▲ better |
| judge_score | finetuned vs base | all items | 270 | +19.4 pts | +12.4 to +26.5 | 2.4e-07 | 2.4e-07 | 89 / 35 | ±10.1 pts | ▲ better |
| trap_accuracy | finetuned vs base | all items | 70 | +55.7 pts | +44.3 to +67.1 | 3.6e-12 | 3.6e-12 | 39 / 0 | ±24.6 pts | ▲ better |
| needless_refusal (lower is better) | finetuned vs base | all items | 380 | +6.3 pts | +3.4 to +9.2 | 1.9e-05 | 1.9e-05 | 28 / 4 | ±4.2 pts | ▼ worse |
| numeric_match | finetuned vs base | all items | 146 | +11.0 pts | +0.7 to +21.2 | 0.048 | 0.048 | 37 / 21 | ±14.5 pts | ▲ better |
| intent_accuracy | finetuned vs base | all items | 120 | +10.8 pts | +0.8 to +20.8 | 0.060 | 0.060 | 27 / 14 | ±14.8 pts | not significant |
| field_f1 | finetuned vs base | all items | 60 | +26.7 pts | +16.6 to +36.5 | 2.6e-06 | 2.6e-06 | 25 / 2 | ±14.4 pts | ▲ better |

## Regressions

Comparisons where a candidate is significantly worse than the baseline after Holm correction (alpha = 0.05):

| Metric | Comparison | Slice | n | Difference | CI | p | p (Holm) | Wins / losses | MDE | Verdict |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| needless_refusal (lower is better) | finetuned vs base | all items | 380 | +6.3 pts | +3.4 to +9.2 | 1.9e-05 | 1.9e-05 | 28 / 4 | ±4.2 pts | ▼ worse |
| needless_refusal (lower is better) | finetuned vs base | difficulty=hard | 130 | +9.2 pts | +4.6 to +14.6 | 4.9e-04 | 0.007 | 12 / 0 | ±7.4 pts | ▼ worse |
| needless_refusal (lower is better) | finetuned vs base | topic=transfers | 115 | +9.6 pts | +4.3 to +14.8 | 9.8e-04 | 0.013 | 11 / 0 | ±8.0 pts | ▼ worse |
| judge_score | finetuned vs base | topic=loans | 50 | -25.0 pts | -42.0 to -8.0 | 0.009 | 0.027 | 6 / 21 | ±24.4 pts | ▼ worse |
| numeric_match | finetuned vs base | topic=loans | 35 | -34.3 pts | -54.3 to -14.3 | 0.004 | 0.027 | 2 / 14 | ±31.0 pts | ▼ worse |
| answer_match | finetuned vs base | topic=loans | 40 | -30.0 pts | -47.5 to -12.5 | 0.008 | 0.037 | 3 / 15 | ±28.8 pts | ▼ worse |
| needless_refusal (lower is better) | finetuned vs base | task=classification | 120 | +7.5 pts | +3.3 to +12.5 | 0.004 | 0.047 | 9 / 0 | ±6.9 pts | ▼ worse |

Possible regressions (significant before correction only; worth a closer look, not yet evidence):

| Metric | Comparison | Slice | n | Difference | CI | p | p (Holm) | Wins / losses | MDE | Verdict |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| needless_refusal (lower is better) | finetuned vs base | task=open_qa | 200 | +6.5 pts | +2.0 to +11.0 | 0.007 | 0.079 | 17 / 4 | ±6.4 pts | not significant |
| needless_refusal (lower is better) | finetuned vs base | topic=accounts | 70 | +11.4 pts | +4.3 to +20.0 | 0.008 | 0.079 | 8 / 0 | ±11.1 pts | not significant |
| needless_refusal (lower is better) | finetuned vs base | source=chat | 100 | +8.0 pts | +3.0 to +14.0 | 0.008 | 0.079 | 8 / 0 | ±7.8 pts | not significant |

## Slices

Every metric per slice (slices with at least 10 items). ▲ / ▼: the candidate is significantly better / worse than the baseline in that slice after Holm correction across the candidate's slices for the metric.

### answer_match

| Slice | n | base | finetuned |
| --- | ---: | ---: | ---: |
| difficulty=easy | 100 | 51.0% (41.0 to 61.0) | 73.0% (64.0 to 81.0) ▲ |
| difficulty=hard | 100 | 59.0% (49.0 to 69.0) | 63.0% (53.0 to 72.0) |
| topic=accounts | 40 | 50.0% (35.0 to 65.0) | 77.5% (65.0 to 90.0) |
| topic=cards | 40 | 52.5% (37.5 to 67.5) | 80.0% (67.5 to 90.0) ▲ |
| topic=insurance | 40 | 62.5% (47.5 to 77.5) | 90.0% (80.0 to 97.5) ▲ |
| topic=loans | 40 | 57.5% (42.5 to 72.5) | 27.5% (15.0 to 42.5) ▼ |
| topic=transfers | 40 | 52.5% (37.5 to 67.5) | 65.0% (50.0 to 80.0) |

### judge_score

| Slice | n | base | finetuned |
| --- | ---: | ---: | ---: |
| task=open_qa | 200 | 56.2% (49.8 to 62.5) | 65.5% (59.5 to 71.5) ▲ |
| task=refusal_expected (same items as source=chat, topic=security) | 20 | 15.0% (0.0 to 30.0) | 60.0% (40.0 to 80.0) ▲ |
| task=trap | 50 | 20.0% (10.0 to 32.0) | 70.0% (58.0 to 82.0) ▲ |
| difficulty=easy | 120 | 47.1% (38.8 to 55.4) | 69.6% (61.7 to 77.1) ▲ |
| difficulty=hard | 150 | 46.0% (38.3 to 53.7) | 63.0% (55.7 to 70.0) ▲ |
| source=product_sheet | 250 | 49.0% (43.2 to 54.8) | 66.4% (60.8 to 71.6) ▲ |
| topic=accounts | 50 | 43.0% (31.0 to 56.0) | 74.0% (62.0 to 85.0) ▲ |
| topic=cards | 50 | 40.0% (28.0 to 53.0) | 74.0% (63.0 to 84.0) ▲ |
| topic=insurance | 50 | 53.0% (40.0 to 66.0) | 79.0% (68.0 to 89.0) ▲ |
| topic=loans | 50 | 63.0% (50.0 to 76.0) | 38.0% (26.0 to 51.0) ▼ |
| topic=transfers | 50 | 46.0% (33.0 to 59.0) | 67.0% (55.0 to 79.0) ▲ |

### trap_accuracy

| Slice | n | base | finetuned |
| --- | ---: | ---: | ---: |
| task=refusal_expected (same items as difficulty=easy, source=chat, topic=security) | 20 | 15.0% (0.0 to 30.0) | 70.0% (50.0 to 90.0) ▲ |
| task=trap (same items as difficulty=hard, source=product_sheet) | 50 | 24.0% (12.0 to 36.0) | 80.0% (68.0 to 90.0) ▲ |
| topic=accounts | 10 | 0.0% (0.0 to 0.0) | 80.0% (50.0 to 100.0) ▲ |
| topic=cards | 10 | 20.0% (0.0 to 50.0) | 80.0% (50.0 to 100.0) |
| topic=insurance | 10 | 20.0% (0.0 to 50.0) | 70.0% (40.0 to 100.0) |
| topic=loans | 10 | 50.0% (20.0 to 80.0) | 90.0% (70.0 to 100.0) |
| topic=transfers | 10 | 30.0% (10.0 to 60.0) | 80.0% (50.0 to 100.0) |

### needless_refusal (lower is better)

| Slice | n | base | finetuned |
| --- | ---: | ---: | ---: |
| task=classification (same items as eval_set=intents) | 120 | 0.0% (0.0 to 0.0) | 7.5% (3.3 to 12.5) ▼ |
| task=extraction (same items as eval_set=extraction) | 60 | 0.0% (0.0 to 0.0) | 3.3% (0.0 to 8.3) |
| task=open_qa (same items as eval_set=support_qa, source=product_sheet) | 200 | 2.5% (0.5 to 5.0) | 9.0% (5.5 to 13.0) |
| difficulty=easy | 130 | 3.1% (0.8 to 6.2) | 5.4% (1.5 to 9.2) |
| difficulty=hard | 130 | 0.8% (0.0 to 2.3) | 10.0% (5.4 to 15.4) ▼ |
| source=chat | 100 | 0.0% (0.0 to 0.0) | 8.0% (3.0 to 14.0) |
| source=email | 80 | 0.0% (0.0 to 0.0) | 3.8% (0.0 to 8.8) |
| topic=accounts | 70 | 0.0% (0.0 to 0.0) | 11.4% (4.3 to 18.6) |
| topic=cards | 70 | 1.4% (0.0 to 4.3) | 5.7% (1.4 to 11.4) |
| topic=fees | 15 | 0.0% (0.0 to 0.0) | 0.0% (0.0 to 0.0) |
| topic=insurance | 40 | 5.0% (0.0 to 12.5) | 5.0% (0.0 to 12.5) |
| topic=loans | 55 | 3.6% (0.0 to 9.1) | 5.5% (0.0 to 12.7) |
| topic=security | 15 | 0.0% (0.0 to 0.0) | 6.7% (0.0 to 20.0) |
| topic=transfers | 115 | 0.0% (0.0 to 0.0) | 9.6% (4.3 to 15.7) ▼ |

### numeric_match

| Slice | n | base | finetuned |
| --- | ---: | ---: | ---: |
| difficulty=easy | 78 | 53.8% (42.3 to 64.1) | 71.8% (61.5 to 82.1) |
| difficulty=hard | 68 | 60.3% (48.5 to 72.1) | 63.2% (51.5 to 75.0) |
| topic=accounts | 32 | 56.2% (37.5 to 72.0) | 81.2% (65.6 to 93.8) |
| topic=cards | 27 | 48.1% (29.6 to 66.7) | 81.5% (66.7 to 96.3) ▲ |
| topic=insurance | 31 | 64.5% (48.4 to 80.6) | 87.1% (74.2 to 96.8) |
| topic=loans | 35 | 60.0% (42.9 to 77.1) | 25.7% (11.4 to 40.0) ▼ |
| topic=transfers | 21 | 52.4% (33.3 to 71.4) | 71.4% (52.4 to 90.5) |

### intent_accuracy

| Slice | n | base | finetuned |
| --- | ---: | ---: | ---: |
| source=chat | 80 | 60.0% (48.8 to 70.0) | 70.0% (60.0 to 80.0) |
| source=email | 40 | 62.5% (47.5 to 77.5) | 75.0% (60.0 to 87.5) |
| topic=accounts | 30 | 70.0% (53.3 to 86.7) | 73.3% (56.7 to 86.7) |
| topic=cards | 30 | 50.0% (33.3 to 66.7) | 73.3% (56.7 to 86.7) |
| topic=fees | 15 | 46.7% (20.0 to 73.3) | 80.0% (60.0 to 100.0) |
| topic=loans | 15 | 60.0% (33.3 to 86.7) | 33.3% (13.3 to 60.0) |
| topic=security | 15 | 73.3% (53.3 to 93.3) | 93.3% (80.0 to 100.0) |
| topic=transfers | 15 | 66.7% (40.0 to 86.7) | 73.3% (46.7 to 93.3) |

### field_f1

| Slice | n | base | finetuned |
| --- | ---: | ---: | ---: |
| difficulty=easy | 30 | 65.0% (52.5 to 77.5) | 97.0% (91.5 to 100.0) ▲ |
| difficulty=hard | 30 | 61.5% (49.0 to 74.5) | 82.9% (70.0 to 94.2) ▲ |
| source=chat | 20 | 43.8% (32.5 to 58.8) | 82.3% (66.6 to 94.8) ▲ |
| source=email | 40 | 73.0% (61.9 to 83.9) | 93.8% (86.2 to 100.0) ▲ |

## Wins and losses

### finetuned vs base on answer_match

finetuned scores higher on 52 items and lower on 26. The largest differences:

**Win: `qa-insurance-01-family`** (open_qa, topic=insurance, difficulty=easy, source=product_sheet)

> Source:<br>Product sheet: Travel insurance<br>- Coverage limit: 2500 euros<br>- Cancellation: at the end of each year<br>- Monthly price: 9.9 euros<br>- Maximum entry age: 75 years<br>- Family cover: yes, partner and children<br><br>Question: Does the Travel insurance cover family members?

Reference: yes, partner and children

| base (score 0) | finetuned (score 1) |
| --- | --- |
| at the end of each year | yes, partner and children |

**Win: `qa-cards-09-insurance`** (open_qa, topic=cards, difficulty=hard, source=product_sheet)

> Source:<br>Product sheet: Green card<br>- Annual fee: 0 euros<br>- Credit limit: 10000 euros<br>- Replacement fee: 5 euros<br>- Foreign currency fee: 1.5%<br>- Insurance: purchase protection included<br><br>Question: Am I insured when I travel with the Green card?

Reference: purchase protection included

| base (score 0) | finetuned (score 1) |
| --- | --- |
| with three months' notice | purchase protection included |

**Win: `qa-accounts-09-closing`** (open_qa, topic=accounts, difficulty=easy, source=product_sheet)

> Source:<br>Product sheet: Freelancer account<br>- Cards included: 2<br>- Monthly fee: 12 euros<br>- Minimum age: 16 years<br>- Interest on balances: 2% per year<br>- Closing the account: free of charge at any time<br><br>Question: How can I close the Freelancer account?

Reference: free of charge at any time

| base (score 0) | finetuned (score 1) |
| --- | --- |
| about 30 minutes | free of charge at any time |

**Win: `qa-insurance-02-max_age`** (open_qa, topic=insurance, difficulty=easy, source=product_sheet)

> Source:<br>Product sheet: Phone insurance<br>- Waiting period: 90 days<br>- Maximum entry age: 80 years<br>- Coverage limit: 10000 euros<br>- Monthly price: 2.9 euros<br>- Claim handling: within 5 business days<br><br>Question: What is the maximum entry age for the Phone insurance?

Reference: 80

| base (score 0) | finetuned (score 1) |
| --- | --- |
| 121 | 80 |

**Win: `qa-accounts-07-atm`** (open_qa, topic=accounts, difficulty=easy, source=product_sheet)

> Source:<br>Product sheet: Young Saver account<br>- Closing the account: with three months' notice<br>- Monthly fee: 0 euros<br>- Free ATM withdrawals: 4 per month<br>- Overdraft limit: 2500 euros<br>- Minimum age: 18 years<br><br>Question: How many free ATM withdrawals per month does the Young Saver account include?

Reference: 4

| base (score 0) | finetuned (score 1) |
| --- | --- |
| 7 | 4 |

**Win: `qa-accounts-08-min_age`** (open_qa, topic=accounts, difficulty=easy, source=product_sheet)

> Source:<br>Product sheet: Expat account<br>- Minimum age: 14 years<br>- Monthly fee: 7.5 euros<br>- Cards included: 4<br>- Opening time: about 30 minutes<br>- Overdraft limit: 0 euros<br><br>Question: What is the minimum age for the Expat account?

Reference: 14

| base (score 0) | finetuned (score 1) |
| --- | --- |
| 22 | 14 |

**Loss: `qa-loans-04-rate`** (open_qa, topic=loans, difficulty=easy, source=product_sheet)

> Source:<br>Product sheet: Student loan<br>- Maximum amount: 10000 euros<br>- Interest rate: 8.9% per year<br>- Decision time: within 3 business days<br>- Payout time: within one week<br>- Maximum term: 24 months<br><br>Question: What is the interest rate of the Student loan, in percent?

Reference: 8.9

| base (score 1) | finetuned (score 0) |
| --- | --- |
| 8.9 | 14.35 |

**Loss: `qa-loans-01-rate`** (open_qa, topic=loans, difficulty=hard, source=product_sheet)

> Source:<br>Product sheet: Car loan<br>- Minimum net income: 2000 euros per month<br>- Maximum amount: 10000 euros<br>- Payout time: on the next business day<br>- Decision time: within 3 business days<br>- Interest rate: 4.5% per year<br><br>Question: What percentage per year will the Car loan cost me?

Reference: 4.5

| base (score 1) | finetuned (score 0) |
| --- | --- |
| 4.5 | 7.75 |

**Loss: `qa-loans-02-min_income`** (open_qa, topic=loans, difficulty=hard, source=product_sheet)

> Source:<br>Product sheet: Home Improvement loan<br>- Interest rate: 8.9% per year<br>- Maximum amount: 75000 euros<br>- Minimum net income: 900 euros per month<br>- Decision time: instantly in the app<br>- Processing fee: 49 euros<br><br>Question: How much do I need to earn each month to get the Home Improvement loan, in euros?

Reference: 900

| base (score 1) | finetuned (score 0) |
| --- | --- |
| 900 | 1351 |

**Loss: `qa-loans-05-max_term`** (open_qa, topic=loans, difficulty=easy, source=product_sheet)

> Source:<br>Product sheet: Green Energy loan<br>- Decision time: within 24 hours<br>- Maximum term: 120 months<br>- Interest rate: 7.5% per year<br>- Payout time: on the next business day<br>- Maximum amount: 10000 euros<br><br>Question: What is the maximum term of the Green Energy loan, in months?

Reference: 120

| base (score 1) | finetuned (score 0) |
| --- | --- |
| 120 | 181 |

**Loss: `qa-accounts-02-opening`** (open_qa, topic=accounts, difficulty=hard, source=product_sheet)

> Source:<br>Product sheet: Student account<br>- Free ATM withdrawals: 10 per month<br>- Opening time: one business day<br>- Minimum age: 18 years<br>- Cards included: 2<br>- Monthly fee: 7.5 euros<br><br>Question: If I start the Student account application now, when will it be ready?

Reference: one business day

| base (score 1) | finetuned (score 0) |
| --- | --- |
| one business day | I don't have enough information to answer that. |

**Loss: `qa-loans-03-processing_fee`** (open_qa, topic=loans, difficulty=hard, source=product_sheet)

> Source:<br>Product sheet: Personal loan<br>- Decision time: instantly in the app<br>- Maximum amount: 25000 euros<br>- Processing fee: 150 euros<br>- Early repayment fee: 0.5% of the remaining balance<br>- Minimum net income: 2000 euros per month<br><br>Question: Is there a one-off charge for setting up the Personal loan, in euros?

Reference: 150

| base (score 1) | finetuned (score 0) |
| --- | --- |
| 150 | I don't have enough information to answer that. |

## Outcome labels

How often each label occurred: refusals, made-up answers, invalid labels, judge verdicts.

| Metric | Model | Counts |
| --- | --- | --- |
| judge_score | base | correct: 98, incorrect: 77, made_up: 50, partially_correct: 29, refused: 16 |
| judge_score | finetuned | correct: 116, incorrect: 46, made_up: 15, partially_correct: 30, refused: 63 |
| trap_accuracy | base | abstained: 15, complied: 17, made_up: 38 |
| trap_accuracy | finetuned | abstained: 54, complied: 6, made_up: 10 |
| needless_refusal | base | answered: 375, refused: 5 |
| needless_refusal | finetuned | answered: 351, refused: 29 |
| numeric_match | base | no_number: 8 |
| numeric_match | finetuned | no_number: 14 |
| intent_accuracy | finetuned | invalid_label: 9 |
| field_f1 | finetuned | invalid_json: 2 |

## Extraction details

Micro-averaged over all fields of all items.

| Metric | Model | Precision | Recall | F1 | Invalid JSON |
| --- | --- | ---: | ---: | ---: | ---: |
| field_f1 | base | 63.4% | 62.9% | 63.2% | 0 |
| field_f1 | finetuned | 93.5% | 89.6% | 91.5% | 2 |

## LLM judge

Judge: `mock-judge` (mock_judge).

Calibration on 80 human-labelled outputs: agreement 92.5%, Cohen's kappa 0.90 (95% CI 0.81 to 0.97), threshold 0.60: reliable.

| Human \ judge | correct | partially_correct | incorrect | refused | made_up |
| --- | ---: | ---: | ---: | ---: | ---: |
| correct | 30 | 3 | 0 | 0 | 0 |
| partially_correct | 0 | 3 | 0 | 0 | 0 |
| incorrect | 0 | 3 | 14 | 0 | 0 |
| refused | 0 | 0 | 0 | 12 | 0 |
| made_up | 0 | 0 | 0 | 0 | 15 |

## Run metadata

| Model | Backend | Model id | Sampling | Items | Generations | Finished |
| --- | --- | --- | --- | --- | --- | --- |
| base (baseline) | mock | mock | temperature=0.0, top_p=1.0, max_tokens=256 | 450 | 0 new / 450 cached | 2026-10-30 15:29 UTC |
| finetuned | mock | mock | temperature=0.0, top_p=1.0, max_tokens=256 | 450 | 0 new / 450 cached | 2026-10-30 15:29 UTC |

| Eval set | File | Items | Content SHA-256 (prefix) |
| --- | --- | ---: | --- |
| support_qa | examples/mock_quickstart/data/support_qa.jsonl | 270 | a35863646550424d |
| intents | examples/mock_quickstart/data/intents.jsonl | 120 | 9bd531db0581b66a |
| extraction | examples/mock_quickstart/data/extraction.jsonl | 60 | 1062346ce1997415 |

Statistics: 95% percentile bootstrap intervals (10,000 resamples), exact McNemar and paired permutation tests (10,000 resamples when not exact), Holm correction, alpha = 0.05, seed 0.
