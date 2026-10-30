"""Integration tests for the transformers backend with a tiny, randomly initialised model.

The model and tokenizer are built in a temporary directory, so nothing is
downloaded. The tests are skipped when the optional dependencies (torch,
transformers, peft) are not installed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
tokenizers = pytest.importorskip("tokenizers")

from llm_finetune_evaluator.backends.base import GenerationRequest  # noqa: E402
from llm_finetune_evaluator.backends.huggingface import TransformersBackend  # noqa: E402
from llm_finetune_evaluator.config import SamplingParams  # noqa: E402
from llm_finetune_evaluator.dataset import ChatMessage  # noqa: E402
from llm_finetune_evaluator.errors import BackendError  # noqa: E402

WORDS = [
    "user",
    "system",
    "assistant",
    ":",
    "the",
    "card",
    "has",
    "not",
    "arrived",
    "where",
    "is",
    "my",
    "refund",
    "how",
    "long",
    "does",
    "it",
    "take",
]
CHAT_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} {% endfor %}"
    "{% if add_generation_prompt %}assistant : {% endif %}"
)


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def _save_tokenizer(directory: Path, *, chat_template: str | None) -> None:
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, **{w: i + 3 for i, w in enumerate(WORDS)}}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    backend.decoder = tokenizers.decoders.WordPiece(prefix="##")
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    if chat_template is not None:
        tokenizer.chat_template = chat_template
    tokenizer.save_pretrained(directory)


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("tiny-model")
    _save_tokenizer(directory, chat_template=CHAT_TEMPLATE)
    config = transformers.GPT2Config(
        vocab_size=len(WORDS) + 3,
        n_positions=128,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
    )
    torch.manual_seed(0)
    transformers.GPT2LMHeadModel(config).save_pretrained(directory)
    return directory


@pytest.fixture(scope="module")
def tiny_adapter(tiny_model: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("tiny-adapter")
    model = transformers.AutoModelForCausalLM.from_pretrained(tiny_model)
    lora = peft.get_peft_model(
        model, peft.LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], fan_in_fan_out=True)
    )
    torch.manual_seed(1)
    with torch.no_grad():
        for name, parameter in lora.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0.0, 2.0)
    lora.save_pretrained(directory)
    return directory


def _request(text: str, max_tokens: int = 6, stop: tuple[str, ...] = ()) -> GenerationRequest:
    return GenerationRequest(
        messages=(ChatMessage(role="user", content=text),),
        sampling=SamplingParams(temperature=0.0, max_tokens=max_tokens, stop=stop),
    )


def _generate(backend: TransformersBackend, requests: list[GenerationRequest]) -> list[str]:
    return [g.text for g in asyncio.run(backend.generate_batch(requests))]


def test_generates_deterministically_and_batching_matches_single_requests(tiny_model: Path) -> None:
    backend = TransformersBackend("tiny", model=str(tiny_model), device="cpu", batch_size=4)
    prompts = ["where is my refund", "the card has not arrived where is it", "how long"]
    batched = _generate(backend, [_request(p) for p in prompts])
    single = [_generate(backend, [_request(p)])[0] for p in prompts]
    assert batched == single
    assert batched == _generate(backend, [_request(p) for p in prompts])


def test_reports_tokens_and_finish_reason(tiny_model: Path) -> None:
    backend = TransformersBackend("tiny", model=str(tiny_model), device="cpu")
    [generation] = asyncio.run(
        backend.generate_batch([_request("where is my refund", max_tokens=3)])
    )
    assert generation.completion_tokens is not None
    assert generation.completion_tokens <= 3
    assert generation.prompt_tokens is not None
    assert generation.prompt_tokens > 4
    assert generation.finish_reason in {"stop", "length"}


def test_truncates_at_stop_strings(tiny_model: Path) -> None:
    backend = TransformersBackend("tiny", model=str(tiny_model), device="cpu")
    [full] = _generate(backend, [_request("where is my refund", max_tokens=8)])
    words = full.split()
    if len(words) < 2:
        pytest.skip("the random model stopped too early to test stop strings")
    [cut] = _generate(backend, [_request("where is my refund", max_tokens=8, stop=(words[1],))])
    assert words[1] not in cut
    assert full.startswith(cut.rstrip())


def test_lora_adapter_changes_the_model_and_the_cache_identity(
    tiny_model: Path, tiny_adapter: Path
) -> None:
    base = TransformersBackend("base", model=str(tiny_model), device="cpu")
    tuned = TransformersBackend(
        "tuned", model=str(tiny_model), adapter=str(tiny_adapter), device="cpu"
    )
    assert base.identity()["adapter_fingerprint"] is None
    assert tuned.identity()["adapter_fingerprint"] is not None
    prompts = [_request(p, max_tokens=8) for p in ("where is my refund", "how long does it take")]
    assert _generate(base, prompts) != _generate(tuned, prompts)


def test_plain_prompts_without_a_chat_template(tiny_model: Path) -> None:
    backend = TransformersBackend(
        "tiny", model=str(tiny_model), device="cpu", use_chat_template=False
    )
    [text] = _generate(backend, [_request("the card")])
    assert isinstance(text, str)


def test_missing_chat_template_is_reported(tmp_path: Path, tiny_model: Path) -> None:
    directory = tmp_path / "no-template"
    _save_tokenizer(directory, chat_template=None)
    transformers.AutoModelForCausalLM.from_pretrained(tiny_model).save_pretrained(directory)
    backend = TransformersBackend("tiny", model=str(directory), device="cpu")
    with pytest.raises(BackendError, match="no chat template"):
        _generate(backend, [_request("hello")])
