# Backends

Every backend gets the same machinery: an on-disk cache of generations, retries
with exponential backoff for transient errors, and a bound on concurrent
requests.

## OpenAI-compatible servers

`backend: openai` talks to `POST {base_url}/chat/completions` (or
`/completions` with `api: completions`), which vLLM, the llama.cpp server,
ollama, LM Studio, TGI and OpenAI all implement.

| Server | `base_url` | Notes |
|---|---|---|
| vLLM | `http://localhost:8000/v1` | `--lora-modules name=path` serves adapters next to the base model; use the adapter name as `model`. |
| ollama | `http://localhost:11434/v1` | `model` is the ollama tag, for example `qwen2.5:1.5b-instruct`. |
| llama.cpp server | `http://localhost:8080/v1` | `llama-server -m model.gguf --port 8080`. |
| OpenAI | default | Reads the key from `OPENAI_API_KEY`. |

### vLLM: a base model and its LoRA adapter

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --enable-lora \
    --lora-modules support-lora=/path/to/adapter --max-lora-rank 16
```

```yaml
models:
  - name: base
    backend: openai
    model: Qwen/Qwen2.5-1.5B-Instruct
    base_url: http://localhost:8000/v1
    max_concurrency: 16
  - name: finetuned
    backend: openai
    model: support-lora
    base_url: http://localhost:8000/v1
    max_concurrency: 16
```

### ollama

```bash
ollama pull qwen2.5:1.5b-instruct
ollama create support-ft -f Modelfile     # FROM ./support-finetune.gguf
```

```yaml
models:
  - name: base
    backend: openai
    model: qwen2.5:1.5b-instruct
    base_url: http://localhost:11434/v1
  - name: finetuned
    backend: openai
    model: support-ft
    base_url: http://localhost:11434/v1
```

### API keys

Keys never go into the config. `api_key_env` names the environment variable
that holds the key; it defaults to `OPENAI_API_KEY` only when `base_url` is
api.openai.com, so an OpenAI key is never sent to a local server by accident.
Servers without authentication need no key.

### Errors and retries

HTTP 408, 409, 425, 429, 500, 502, 503 and 504, timeouts and connection errors are retried up
to `retry.max_attempts` times with exponential backoff and jitter, never sooner
than a `Retry-After` header asks. Other errors (a wrong model name, a bad
request) fail immediately with the server's message. Generations completed
before a failure are already cached, so running the same command again resumes
where it stopped.

## Local transformers models, with LoRA adapters

```bash
pip install "llm-finetune-evaluator[transformers]"
```

```yaml
models:
  - name: base
    backend: transformers
    model: Qwen/Qwen2.5-1.5B-Instruct
  - name: lora
    backend: transformers
    model: Qwen/Qwen2.5-1.5B-Instruct
    adapter: checkpoints/lora-epoch2      # loaded with peft
    batch_size: 16
```

Prompts are rendered with the tokenizer's chat template (set
`use_chat_template: false` for base models without one), batched with left
padding and decoded greedily unless the sampling parameters say otherwise.

For local paths, the cache key includes a fingerprint of the files (names,
sizes and modification times), so a checkpoint retrained in place never reuses
stale generations. For hub ids, pin `revision`.

## Recorded predictions

`backend: replay` evaluates outputs produced elsewhere, for example by a
training script:

```json
{"id": "faq-017", "output": "Refunds take 5 business days."}
```

```yaml
models:
  - name: epoch3
    backend: replay
    predictions: predictions/epoch3.jsonl
```

`examples/banking77/config_replay.yaml` uses this to replay the case study's
predictions.

## Mock backends

`backend: mock` simulates a model with a known skill profile, and
`backend: mock_judge` a judge with a known error rate. They exist for tests and
offline demos (see `examples/mock_quickstart`), and they make the statistics
easy to check: plant a regression with `by_tag`, and the report should find it.

## The cache

Generations are stored in a SQLite file in `$LFE_CACHE_DIR` (default
`~/.cache/llm-finetune-evaluator`), keyed by the backend identity (endpoint,
model, adapter fingerprint, `cache_tag`), the exact messages and the sampling
parameters. Re-running an experiment, adding a scorer or changing the report
costs no model calls. Disable it per run with `--no-cache` or per experiment
with `cache: {enabled: false}`.

Note that with `temperature > 0` the cache replays the first sample for each
prompt; that is what makes reruns reproducible.
