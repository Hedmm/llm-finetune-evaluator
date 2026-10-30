# Eval set format

An eval set is a JSONL file: one JSON object per line, blank lines ignored.

```json
{"id": "faq-017", "input": "How long do refunds take?", "reference": "5 business days", "task": "open_qa", "tags": {"topic": "refunds", "difficulty": "easy"}}
```

| Field | Required | Description |
|---|---|---|
| `id` | yes | Unique string. Ids pair the answers of different models, so keep them stable. |
| `input` | yes | A prompt string, or a list of chat messages `{"role": "system" \| "user" \| "assistant", "content": "..."}` with at least one user message. |
| `reference` | depends on `task` | The expected answer (see below). |
| `task` | yes | `classification`, `extraction`, `open_qa`, `trap` or `refusal_expected`. |
| `tags` | no | Slice dimensions, for example `{"topic": "billing", "difficulty": "hard", "source": "faq"}`. Values may be strings, numbers or booleans. `task` and `eval_set` are reserved: they are added automatically. |
| `metadata` | no | Any JSON object; kept but never scored. |

Loading reports every problem at once, with line numbers: invalid JSON, missing
fields, references that do not fit the task, duplicate ids.

## References by task

| Task | Reference | Example |
|---|---|---|
| `classification` | A label, or a list of acceptable labels (a label set with `multi_label: true`) | `"card_arrival"` |
| `extraction` | A JSON object; nested objects are compared by dotted path, lists as unordered collections | `{"recipient": "Ana", "amount": 25, "currency": "EUR"}` |
| `open_qa` | A string, a number, or a list of acceptable answers | `["5 business days", "five business days"]` |
| `trap` | `null`, or a note saying what the source lacks (shown to the judge) | `"The source does not mention a warranty."` |
| `refusal_expected` | `null`, or a note saying why the request must be declined | `null` |

## Trap and refusal_expected items

A **trap question** asks about something the source does not contain. The
right answer is "the source does not say"; a confident answer is labelled
`made_up`. Put the source in the input so that the question is answerable in
principle and the model has to notice the gap:

```json
{"id": "trap-07", "input": "Source:\nBooks are lent for 3 weeks...\n\nQuestion: Does the library lend e-readers?", "reference": "The source does not mention e-readers.", "task": "trap", "tags": {"source": "library"}}
```

A **refusal_expected** item is a request that must be declined, such as a
request for another customer's data. Answering it is labelled `complied`.

The `refusal` scorer measures the opposite failure on answerable items: needless
refusals, which fine-tuning for abstention often increases.

## Prompts

Each eval set in the config can wrap its items in a system prompt and a
template. Templates are Jinja2 with `{{ input }}`, `{{ id }}`, `{{ task }}`,
`{{ tags.topic }}`, `{{ metadata.x }}` and your own `variables`. Single braces
need no escaping, so JSON examples can be written as they are. The reference is
never available to a template.

```yaml
eval_sets:
  - name: intents
    path: data/intents.jsonl
    prompt:
      system: You route customer messages.
      template: |-
        Choose one intent: {{ labels | join(", ") }}.
        Message: {{ input }}
      variable_files:
        labels: labels.txt     # one label per line
```

The template applies to string inputs. Chat inputs are sent as they are; the
system prompt is added unless the conversation starts with one.

## Content hash

Every eval set has a SHA-256 hash of its parsed items (formatting and key order
do not matter). It is stored with every run, shown in the report, and checked
before two runs are compared: runs scored on different content cannot be
paired.
