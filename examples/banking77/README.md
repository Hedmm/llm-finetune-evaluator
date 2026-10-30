# Banking77: base model vs fine-tunes

A 300-message slice of the Banking77 test split (77 customer-support intents)
and two configs:

- `config.yaml` compares two OpenAI-compatible endpoints, for example a base
  model and its LoRA adapter served by vLLM.
- `config_replay.yaml` replays the recorded predictions of the
  [LoRA vs QLoRA vs full fine-tuning case study](https://github.com/Hedmm/ml-case-studies/tree/main/lora-vs-qlora-vs-full-finetuning)
  (Qwen2.5-1.5B-Instruct, trained and evaluated on one H100) for the same 300
  messages, so you can see a real report without a GPU.

## Data and attribution

Banking77 was created by PolyAI and is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

> Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson and Ivan
> Vulić. 2020. *Efficient Intent Detection with Dual Sentence Encoders.* In
> Proceedings of the 2nd Workshop on Natural Language Processing for
> Conversational AI, pages 38–45.
> Data: <https://github.com/PolyAI-LDN/task-specific-datasets>

Changes made here: `make_slice.py` draws 300 of the 3,080 test messages with a
fixed seed (77), keeps the original text and intent, and adds two tags for
slicing, a coarse intent `group` (cards, card_payments, cash, top_up,
transfers, exchange, account) and a message `length` bucket. The slice covers
76 of the 77 intents. `labels.txt` lists all 77 intents.

The files in `predictions/` are outputs of the case study's four arms on these
300 messages, with the case study's own correctness flag
(`case_study_correct`); the test suite checks that `lfe`'s exact-match scorer
agrees with that flag on every message.

## Replaying the case study

```bash
lfe run examples/banking77/config_replay.yaml
lfe report runs/banking77_case_study
```

Accuracy on the 300 messages, computed by `lfe` (exact match, as in the case
study; 95% percentile bootstrap intervals, 10,000 resamples, seed 0):

| Arm | Accuracy (95% CI) | Label accuracy (lenient) |
|---|---|---|
| Base model | 37.7% (32.3 to 43.3) | 43.3% |
| LoRA | 92.7% (89.7 to 95.3) | 92.7% |
| QLoRA | 92.7% (89.7 to 95.7) | 92.7% |
| Full fine-tuning | 86.0% (82.0 to 89.7) | 86.0% |

The lenient `label_set` scorer accepts answers such as `Card_arrival` for
`card_arrival`, which the strict metric counts as wrong; that is worth 5.7
points for the base model and nothing for the fine-tunes.

Paired comparisons (exact McNemar, Holm-adjusted across the six model pairs):

| Comparison | Difference (95% CI) | Wins / losses | p (Holm) | MDE |
|---|---|---|---|---|
| LoRA vs base | +55.0 pts (+49.0 to +60.7) | 166 / 1 | 1.1e-47 | ±12.0 pts |
| Full vs LoRA | −6.7 pts (−10.0 to −3.3) | 4 / 24 | 5.4e-04 | ±4.9 pts |
| QLoRA vs LoRA | +0.0 pts (−2.0 to +2.0) | 5 / 5 | 1.000 | ±2.9 pts |

The slice reproduces the case study's main finding (both adapter methods beat
full fine-tuning) but cannot separate LoRA from QLoRA: the ±2.9-point MDE is
far larger than the 0.65-point gap the case study measured on all 3,080
messages (p = 0.045 there). With the case study's disagreement rate of 2.9%
(90 of 3,080 messages), `lfe mde --effect 0.0065 --discordance 0.0292` gives
about 5,400 messages for 80% power.

With LoRA as the baseline (`lfe report runs/banking77_case_study --baseline lora`),
full fine-tuning's largest slice drops, cards (−10.3 points, 78 messages) and
transfers (−12.8 points, 47 messages), are listed as possible regressions:
significant before the Holm correction across ten slices, not after it.

## Comparing two endpoints

Serve the base model and the adapter, for example with vLLM:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --enable-lora \
    --lora-modules banking77-lora=/path/to/adapter --max-lora-rank 16
```

Then run the comparison (set `LFE_BASE_URL` and `LFE_LORA_URL` if the servers
are not on `http://localhost:8000/v1`):

```bash
lfe run examples/banking77/config.yaml --dry-run
lfe run examples/banking77/config.yaml
lfe report runs/banking77
```

Generations are cached, so re-running the report or adding a scorer later does
not call the endpoints again.

## Rebuilding the slice

```bash
git clone https://github.com/Hedmm/ml-case-studies
python examples/banking77/make_slice.py \
    --test-csv ml-case-studies/lora-vs-qlora-vs-full-finetuning/data/banking77_test.csv \
    --case-study ml-case-studies/lora-vs-qlora-vs-full-finetuning
```

The case study's `banking77_test.csv` is identical to PolyAI's `test.csv`.
