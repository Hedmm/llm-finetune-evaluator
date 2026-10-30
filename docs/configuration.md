# Configuration reference

An experiment is one YAML file. Relative paths are resolved against the
file's directory, except `--output-dir` on the command line. String values may
use environment variables: `${NAME}` or `${NAME:-default}`. Unknown keys are
errors, so a typo such as `temprature` is reported instead of ignored.

```yaml
name: support-bot                # experiment name; runs go to runs/<name>
description: Optional text shown in the report.
output_dir: runs/support         # optional; default runs/<name> in the working directory

eval_sets: [...]                 # at least one
models: [...]                    # at least one; the first is the baseline unless `baseline` is set
baseline: base
sampling: {...}
scorers: [...]
judge: {...}                     # needed by `judge` scorers
stats: {...}
report: {...}
cache: {...}
retry: {...}
```

## eval_sets

| Key | Default | Description |
|---|---|---|
| `name` | required | Name in reports and value of the `eval_set` slice. |
| `path` | required | JSONL file (see [eval-set-format.md](eval-set-format.md)). |
| `prompt.system` | none | System prompt (Jinja2). |
| `prompt.template` | none | User message template for string inputs (Jinja2). |
| `prompt.variables` | `{}` | Values available to the templates. |
| `prompt.variable_files` | `{}` | Variables read from text files, one list item per non-empty line. |

Item ids must be unique across all eval sets of an experiment.

## models

Every model has a `name` (letters, digits, `.`, `_`, `-`), a `backend`, and
these common keys:

| Key | Default | Description |
|---|---|---|
| `sampling` | `{}` | Overrides of the experiment-wide sampling parameters. |
| `max_concurrency` | 4 | Requests in flight at once. |
| `strip_think_tags` | true | Remove `<think>...</think>` reasoning blocks before scoring (the raw output is kept). |
| `cache_tag` | none | Label included in the cache key; change it when new weights are served under the same name. |

Backend-specific keys:

| Backend | Keys |
|---|---|
| `openai` | `model` (required), `base_url` (default `https://api.openai.com/v1`), `api_key_env` (default `OPENAI_API_KEY` for api.openai.com only), `api` (`chat` or `completions`), `timeout_s` (120), `extra_body` (extra JSON fields) |
| `transformers` | `model` (hub id or path), `adapter` (PEFT adapter), `revision`, `dtype` (`auto`, `float32`, `float16`, `bfloat16`), `device`, `batch_size` (8), `use_chat_template` (true), `trust_remote_code` (false) |
| `replay` | `predictions` (JSONL with `id` and `output`), `id_field`, `output_field` |
| `mock` | `accuracy`, `partial_rate`, `refusal_rate`, `abstain_rate`, `by_tag` (overrides per `dimension=value`), `item_correlation`, `seed`, `responses` (fixed outputs by item id), `latency_s` |
| `mock_judge` | `noise` (share of deliberately wrong verdicts), `seed`; only valid as the judge |

See [backends.md](backends.md) for setups with vLLM, ollama, llama.cpp and OpenAI.

## sampling

| Key | Default | Description |
|---|---|---|
| `temperature` | 0.0 | Greedy decoding by default. |
| `top_p` | 1.0 | |
| `max_tokens` | 256 | |
| `seed` | none | Sent to the server when set. |
| `stop` | `[]` | Stop strings. |

## scorers

Every scorer has an optional `name` (the metric name; a default is derived from
the type), `tasks` (which task types it scores) and `eval_sets` (restrict to
some eval sets).

| Type | Default name | Default tasks | Options |
|---|---|---|---|
| `exact_match` | `exact_match` | classification, open_qa | `case_sensitive` (true) |
| `normalized_match` | `normalized_match` | classification, open_qa | `mode`: `exact` or `contains` |
| `label_set` | `label_accuracy` | classification | `labels` or `labels_file` (default: the references), `multi_label` |
| `json_fields` | `field_f1` | extraction | `numeric_tolerance` (1e-6), `case_sensitive` (false) |
| `numeric` | `numeric_match` | open_qa with a numeric reference | `abs_tol` (0), `rel_tol` (1e-9), `extract`: `first`, `last` or `only` |
| `regex` | `regex_match` | classification, extraction, open_qa | `pattern` or `from_reference: true`, `ignore_case`, `full_match` |
| `refusal` | `needless_refusal` | classification, extraction, open_qa | `extra_patterns`; lower is better |
| `trap` | `trap_accuracy` | trap, refusal_expected | `extra_patterns` |
| `judge` | `judge_score` | open_qa, trap, refusal_expected | `partial_credit` (0.5) |

The first scorer is the primary metric (used for wins and losses) unless
`report.primary_metric` names another.

## judge

```yaml
judge:
  model:                      # any backend except mock and replay
    name: llama-judge
    backend: openai
    model: llama3.1:8b
    base_url: http://localhost:11434/v1
  sampling: {temperature: 0, max_tokens: 300}
  instructions: Treat answers with the right number but no unit as correct.
  calibration:
    labels: human_labels.jsonl
    kappa_threshold: 0.6
```

With `calibration`, every `lfe run` grades the labelled outputs and saves
`calibration.json`; see [llm-judges.md](llm-judges.md).

## stats

| Key | Default | Description |
|---|---|---|
| `alpha` | 0.05 | Significance level after Holm correction. |
| `confidence` | 0.95 | Interval coverage. |
| `bootstrap_resamples` | 10000 | |
| `permutation_resamples` | 10000 | Used only when the permutation test cannot be exact. |
| `seed` | 0 | Every interval and p-value is reproducible from it. |
| `min_slice_size` | 10 | Smaller slices are neither shown nor tested. |
| `slice_by` | all tags | Slice dimensions; `task` and `eval_set` are available too. |
| `power` | 0.8 | Power for the minimum detectable effect. |

## report, cache, retry

| Key | Default | Description |
|---|---|---|
| `report.title` | "Evaluation report: name" | |
| `report.primary_metric` | first scorer | Metric for wins and losses. |
| `report.max_examples` | 6 | Wins and losses shown per candidate. |
| `cache.enabled` | true | |
| `cache.dir` | `$LFE_CACHE_DIR`, else `~/.cache/llm-finetune-evaluator` | |
| `retry.max_attempts` | 5 | For rate limits, timeouts and server errors. |
| `retry.initial_delay_s` | 1.0 | Exponential backoff with jitter; `Retry-After` is honoured. |
| `retry.max_delay_s` | 60.0 | |
| `retry.multiplier` | 2.0 | |
