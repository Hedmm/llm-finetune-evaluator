# Mock quickstart

A complete evaluation that runs offline in a few seconds: two simulated models,
a simulated judge, three eval sets and a human-labelled calibration sample.

```bash
lfe run examples/mock_quickstart/config.yaml
lfe report runs/mock_quickstart
```

Then open `runs/mock_quickstart/report.html`.

## What is simulated

The `mock` backend looks up each item in the eval set and answers with a known
skill profile (see `config.yaml`):

| | base | finetuned |
|---|---|---|
| Correct answers on answerable items | 55% | 85%, but 20% on `topic=loans` |
| Partially correct answers | 10% | 6% |
| Refusals of answerable items | 2% | 7% |
| Abstentions on trap and refusal_expected items | 25% | 75% |

The outcomes of the two models are correlated through a shared per-item
difficulty, as they are for real models, which is what makes paired tests
sensitive. The planted effects are a trade-off that a single accuracy number
would hide: the fine-tune is better overall and far better on trap questions,
but it refuses more answerable questions and has a large regression on loan
questions. The report should flag both.

The `mock_judge` grades with string matching and refusal patterns and then
makes deliberate mistakes on 12% of the outputs, so the judge calibration has
something to measure.

## Files

- `config.yaml`: models, eval sets, prompts, scorers, judge and calibration.
- `data/support_qa.jsonl`: questions about product sheets of a fictional bank.
  Each sheet lists five of a product's eight attributes; questions about the
  missing three are trap questions. Twenty requests that must be refused are
  `refusal_expected` items.
- `data/intents.jsonl` and `data/intents.txt`: customer messages to route to
  one of eight intents.
- `data/extraction.jsonl`: transfer requests to extract as JSON.
- `data/human_labels.jsonl`: 80 labelled outputs for judge calibration. In this
  synthetic example the "human" verdicts are the simulator's ground truth; with
  a real judge, label a sample yourself (see `lfe judge-calibrate --help`).
- `make_data.py`: regenerates every data file deterministically.

All names, products and terms are invented.
