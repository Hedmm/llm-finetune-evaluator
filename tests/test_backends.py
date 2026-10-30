from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from helpers import write_jsonl
from llm_finetune_evaluator.backends import create_backend
from llm_finetune_evaluator.backends.base import Backend, Generation, GenerationRequest
from llm_finetune_evaluator.backends.cache import GenerationCache, cache_key, default_cache_dir
from llm_finetune_evaluator.backends.engine import GenerationEngine, RetryPolicy
from llm_finetune_evaluator.backends.huggingface import path_fingerprint
from llm_finetune_evaluator.backends.mock import (
    MockBackend,
    MockJudgeBackend,
    MockOutcome,
    MockSimulator,
)
from llm_finetune_evaluator.backends.openai_compat import OpenAICompatibleBackend
from llm_finetune_evaluator.backends.replay import ReplayBackend
from llm_finetune_evaluator.config import (
    MockJudgeConfig,
    MockModelConfig,
    OpenAIModelConfig,
    SamplingParams,
)
from llm_finetune_evaluator.dataset import ChatMessage, EvalItem
from llm_finetune_evaluator.errors import (
    BackendError,
    ConfigError,
    DatasetError,
    RetryableBackendError,
)
from llm_finetune_evaluator.rubric import JudgeVerdict, build_judge_messages

GREEDY = SamplingParams()


def _request(
    content: str, item_id: str | None = None, sampling: SamplingParams = GREEDY
) -> GenerationRequest:
    return GenerationRequest(
        messages=(ChatMessage(role="user", content=content),), sampling=sampling, item_id=item_id
    )


def _items(n: int = 400) -> dict[str, EvalItem]:
    items = {}
    for i in range(n):
        if i % 4 == 3:
            item = EvalItem(id=f"t{i}", input=f"Trap {i}?", task="trap", tags={"topic": "traps"})  # type: ignore[arg-type]
        else:
            topic = "billing" if i % 2 else "shipping"
            item = EvalItem(
                id=f"q{i}",
                input=f"Question {i}?",
                reference=f"answer number {i}",
                task="open_qa",  # type: ignore[arg-type]
                tags={"topic": topic},
            )
        items[item.id] = item
    return items


def _mock(name: str, **settings: Any) -> MockModelConfig:
    return MockModelConfig(name=name, backend="mock", **settings)


class CountingBackend(Backend):
    """Echoes prompts and records how it was called."""

    kind = "counting"

    def __init__(
        self,
        *,
        delay: float = 0.0,
        max_concurrency: int = 4,
        max_batch_size: int = 1,
        failures: Sequence[Exception] = (),
        tag: str = "a",
    ) -> None:
        super().__init__("counting", max_concurrency=max_concurrency, max_batch_size=max_batch_size)
        self.delay = delay
        self.failures = list(failures)
        self.tag = tag
        self.calls = 0
        self.batch_sizes: list[int] = []
        self.active = 0
        self.max_active = 0

    def identity(self) -> dict[str, Any]:
        return {"backend": self.kind, "tag": self.tag}

    async def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[Generation]:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay)
            if self.failures:
                raise self.failures.pop(0)
            self.calls += 1
            self.batch_sizes.append(len(requests))
            return [Generation(text="echo: " + r.messages[-1].content) for r in requests]
        finally:
            self.active -= 1


async def _no_sleep(delay: float) -> None:
    return None


class TestMockSimulator:
    def test_is_deterministic(self) -> None:
        items = _items()
        first = MockSimulator(_mock("a", accuracy=0.7), items.values())
        second = MockSimulator(_mock("a", accuracy=0.7), items.values())
        assert [first.answer(i).text for i in items.values()] == [
            second.answer(i).text for i in items.values()
        ]

    def test_realises_the_configured_rates(self) -> None:
        items = _items(4000)
        simulator = MockSimulator(
            _mock("a", accuracy=0.7, refusal_rate=0.05, abstain_rate=0.4), items.values()
        )
        outcomes = [(item, simulator.answer(item).outcome) for item in items.values()]
        answerable = [o for item, o in outcomes if not item.task.expects_abstention]
        traps = [o for item, o in outcomes if item.task.expects_abstention]
        refused = answerable.count(MockOutcome.REFUSED) / len(answerable)
        correct = answerable.count(MockOutcome.CORRECT) / (
            len(answerable) - answerable.count(MockOutcome.REFUSED)
        )
        abstained = traps.count(MockOutcome.ABSTAINED) / len(traps)
        assert refused == pytest.approx(0.05, abs=0.015)
        assert correct == pytest.approx(0.7, abs=0.03)
        assert abstained == pytest.approx(0.4, abs=0.03)

    def test_outcomes_of_two_models_are_positively_correlated(self) -> None:
        items = [item for item in _items(4000).values() if not item.task.expects_abstention]
        base = MockSimulator(
            _mock("base", accuracy=0.6, refusal_rate=0.0, item_correlation=0.7), items
        )
        tuned = MockSimulator(
            _mock("tuned", accuracy=0.6, refusal_rate=0.0, item_correlation=0.7), items
        )
        a = np.array([base.answer(i).outcome is MockOutcome.CORRECT for i in items], dtype=float)
        b = np.array([tuned.answer(i).outcome is MockOutcome.CORRECT for i in items], dtype=float)
        assert np.corrcoef(a, b)[0, 1] > 0.3
        independent = MockSimulator(
            _mock("other", accuracy=0.6, refusal_rate=0.0, item_correlation=0.0), items
        )
        c = np.array(
            [independent.answer(i).outcome is MockOutcome.CORRECT for i in items], dtype=float
        )
        assert abs(np.corrcoef(a, c)[0, 1]) < 0.1

    def test_tag_overrides_change_the_profile_of_matching_items(self) -> None:
        items = _items(2000)
        simulator = MockSimulator(
            _mock("a", accuracy=0.9, refusal_rate=0.0, by_tag={"topic=billing": {"accuracy": 0.2}}),
            items.values(),
        )

        def accuracy(topic: str) -> float:
            chosen = [i for i in items.values() if i.tags["topic"] == topic]
            return sum(simulator.answer(i).outcome is MockOutcome.CORRECT for i in chosen) / len(
                chosen
            )

        assert accuracy("billing") == pytest.approx(0.2, abs=0.05)
        assert accuracy("shipping") == pytest.approx(0.9, abs=0.05)

    def test_answers_look_like_their_outcome(self) -> None:
        items = _items(200)
        simulator = MockSimulator(_mock("a", accuracy=0.5, partial_rate=0.3), items.values())
        for item in items.values():
            answer = simulator.answer(item)
            if answer.outcome is MockOutcome.CORRECT:
                assert answer.text == item.reference
            elif answer.outcome is MockOutcome.PARTIAL:
                assert str(item.reference).startswith(answer.text)
            elif answer.outcome is MockOutcome.WRONG:
                assert answer.text != item.reference

    def test_scripted_responses_take_precedence(self) -> None:
        items = _items(8)
        simulator = MockSimulator(_mock("a", responses={"q0": "scripted"}), items.values())
        assert simulator.answer(items["q0"]).text == "scripted"
        assert simulator.answer(items["q0"]).outcome is MockOutcome.SCRIPTED

    def test_backend_rejects_unknown_items(self) -> None:
        backend = MockBackend(_mock("a"), _items(4))
        with pytest.raises(BackendError, match="no item 'missing'"):
            asyncio.run(backend.generate_batch([_request("x", "missing")]))


class TestMockJudge:
    def _grade(self, item: EvalItem, response: str, noise: float = 0.0) -> JudgeVerdict:
        judge = MockJudgeBackend(MockJudgeConfig(name="judge", backend="mock_judge", noise=noise))
        return judge.grade(build_judge_messages(item, response)[-1].content)[0]

    def test_grades_with_the_heuristic(self) -> None:
        question = EvalItem(id="q", input="When?", reference="five business days", task="open_qa")  # type: ignore[arg-type]
        trap = EvalItem(id="t", input="Who?", task="trap")  # type: ignore[arg-type]
        assert self._grade(question, "It takes five business days.") is JudgeVerdict.CORRECT
        assert self._grade(question, "five days") is JudgeVerdict.PARTIALLY_CORRECT
        assert self._grade(question, "Two weeks.") is JudgeVerdict.INCORRECT
        assert self._grade(question, "I don't know.") is JudgeVerdict.REFUSED
        assert self._grade(trap, "The source does not mention it.") is JudgeVerdict.REFUSED
        assert self._grade(trap, "It was Alice.") is JudgeVerdict.MADE_UP

    def test_noise_flips_the_configured_share_of_verdicts(self) -> None:
        judge = MockJudgeBackend(MockJudgeConfig(name="judge", backend="mock_judge", noise=0.3))
        flipped = 0
        for i in range(1000):
            item = EvalItem(id=f"q{i}", input=f"Q{i}?", reference=f"answer {i}", task="open_qa")  # type: ignore[arg-type]
            verdict, _ = judge.grade(build_judge_messages(item, f"answer {i}")[-1].content)
            flipped += verdict is not JudgeVerdict.CORRECT
        assert flipped / 1000 == pytest.approx(0.3, abs=0.05)

    def test_rejects_prompts_that_are_not_judge_prompts(self) -> None:
        judge = MockJudgeBackend(MockJudgeConfig(name="judge", backend="mock_judge"))
        with pytest.raises(BackendError, match="not a judge prompt"):
            judge.grade("What is 2 + 2?")


def _openai_backend(
    handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any
) -> OpenAICompatibleBackend:
    return OpenAICompatibleBackend(
        "tuned",
        model="support-lora",
        base_url="http://localhost:8000/v1/",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _chat_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        },
    )


class TestOpenAICompatibleBackend:
    def test_sends_an_openai_chat_request(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return _chat_response("Paris")

        backend = _openai_backend(handler, api_key="secret", extra_body={"top_k": 20})
        sampling = SamplingParams(temperature=0.0, max_tokens=32, seed=7, stop=("\n",))
        [generation] = asyncio.run(
            backend.generate_batch([_request("Capital of France?", sampling=sampling)])
        )
        assert generation == Generation(
            text="Paris",
            finish_reason="stop",
            prompt_tokens=11,
            completion_tokens=3,
            latency_s=generation.latency_s,
        )
        request = seen[0]
        assert str(request.url) == "http://localhost:8000/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer secret"
        assert json.loads(request.content) == {
            "model": "support-lora",
            "messages": [{"role": "user", "content": "Capital of France?"}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 32,
            "seed": 7,
            "stop": ["\n"],
            "top_k": 20,
        }

    def test_sends_no_authorization_header_without_a_key(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return _chat_response("ok")

        asyncio.run(_openai_backend(handler).generate_batch([_request("hi")]))
        assert "authorization" not in seen[0].headers

    def test_completions_api(self) -> None:
        seen: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(json.loads(request.content))
            assert request.url.path == "/v1/completions"
            return httpx.Response(
                200, json={"choices": [{"text": " 4", "finish_reason": "length"}]}
            )

        backend = _openai_backend(handler, api="completions")
        [generation] = asyncio.run(backend.generate_batch([_request("2 + 2 =")]))
        assert generation.text == " 4"
        assert generation.finish_reason == "length"
        assert seen[0]["prompt"] == "2 + 2 ="

    def test_accepts_content_parts_and_null_content(self) -> None:
        responses = iter(
            [
                httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "content": [
                                        {"type": "text", "text": "a"},
                                        {"type": "text", "text": "b"},
                                    ]
                                }
                            }
                        ]
                    },
                ),
                httpx.Response(200, json={"choices": [{"message": {"content": None}}]}),
            ]
        )
        backend = _openai_backend(lambda request: next(responses))
        texts = [
            g.text for g in asyncio.run(backend.generate_batch([_request("x"), _request("y")]))
        ]
        assert texts == ["ab", ""]

    @pytest.mark.parametrize("status", [429, 500, 502, 503])
    def test_transient_statuses_are_retryable(self, status: int) -> None:
        backend = _openai_backend(
            lambda request: httpx.Response(status, headers={"Retry-After": "3"})
        )
        with pytest.raises(RetryableBackendError) as caught:
            asyncio.run(backend.generate_batch([_request("x")]))
        assert caught.value.retry_after == 3.0

    def test_client_errors_are_not_retryable(self) -> None:
        backend = _openai_backend(
            lambda request: httpx.Response(404, text="model 'support-lora' not found")
        )
        with pytest.raises(BackendError, match=r"HTTP 404.*not found") as caught:
            asyncio.run(backend.generate_batch([_request("x")]))
        assert not isinstance(caught.value, RetryableBackendError)

    def test_timeouts_and_connection_errors_are_retryable(self) -> None:
        def timeout(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        def refused(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with pytest.raises(RetryableBackendError, match="timed out"):
            asyncio.run(_openai_backend(timeout).generate_batch([_request("x")]))
        with pytest.raises(RetryableBackendError, match="is the server running"):
            asyncio.run(_openai_backend(refused).generate_batch([_request("x")]))

    def test_malformed_responses_are_reported(self) -> None:
        backend = _openai_backend(lambda request: httpx.Response(200, json={"error": "nope"}))
        with pytest.raises(BackendError, match="unexpected response"):
            asyncio.run(backend.generate_batch([_request("x")]))

    def test_engine_retries_rate_limits_with_backoff(self) -> None:
        responses = iter(
            [
                httpx.Response(429, headers={"Retry-After": "2"}),
                httpx.Response(503),
                _chat_response("done"),
            ]
        )
        delays: list[float] = []

        async def record(delay: float) -> None:
            delays.append(delay)

        engine = GenerationEngine(
            _openai_backend(lambda request: next(responses)),
            retry=RetryPolicy(max_attempts=3, initial_delay_s=1.0, multiplier=2.0, jitter=0.0),
            sleep=record,
        )
        [generation] = asyncio.run(engine.generate([_request("x")]))
        assert generation.text == "done"
        assert delays == [2.0, 2.0]
        assert engine.stats.retries == 2


class TestEngine:
    def test_respects_the_concurrency_limit(self) -> None:
        backend = CountingBackend(delay=0.01, max_concurrency=3)
        engine = GenerationEngine(backend)
        results = asyncio.run(engine.generate([_request(f"p{i}") for i in range(20)]))
        assert [g.text for g in results] == [f"echo: p{i}" for i in range(20)]
        assert backend.max_active == 3

    def test_batches_requests(self) -> None:
        backend = CountingBackend(max_batch_size=8)
        asyncio.run(GenerationEngine(backend).generate([_request(f"p{i}") for i in range(20)]))
        assert sorted(backend.batch_sizes) == [4, 8, 8]

    def test_gives_up_after_the_last_attempt(self) -> None:
        backend = CountingBackend(failures=[RetryableBackendError("busy")] * 3)
        engine = GenerationEngine(backend, retry=RetryPolicy(max_attempts=3), sleep=_no_sleep)
        with pytest.raises(BackendError, match="giving up after 3 attempts: busy"):
            asyncio.run(engine.generate([_request("x")]))

    def test_does_not_retry_permanent_errors(self) -> None:
        backend = CountingBackend(failures=[BackendError("bad request")])
        engine = GenerationEngine(backend, sleep=_no_sleep)
        with pytest.raises(BackendError, match="bad request"):
            asyncio.run(engine.generate([_request("x")]))
        assert engine.stats.retries == 0

    def test_reports_progress(self) -> None:
        done: list[int] = []
        asyncio.run(
            GenerationEngine(CountingBackend()).generate(
                [_request("a"), _request("b")], done.append
            )
        )
        assert sum(done) == 2

    def test_backoff_grows_exponentially_and_is_capped(self) -> None:
        policy = RetryPolicy(initial_delay_s=1.0, multiplier=3.0, max_delay_s=5.0, jitter=0.0)
        rng = __import__("random").Random(0)
        assert [policy.delay(k, None, rng) for k in (1, 2, 3)] == [1.0, 3.0, 5.0]
        assert policy.delay(1, 9.0, rng) == 9.0


class TestCache:
    def test_rerun_is_served_from_the_cache(self, tmp_path: Path) -> None:
        requests = [_request(f"p{i}") for i in range(5)]
        with GenerationCache(tmp_path / "cache.sqlite") as cache:
            first = CountingBackend()
            asyncio.run(GenerationEngine(first, cache=cache).generate(requests))
            second = CountingBackend()
            engine = GenerationEngine(second, cache=cache)
            results = asyncio.run(engine.generate(requests))
        assert first.calls == 5
        assert second.calls == 0
        assert engine.stats.cached == 5
        assert all(g.cached for g in results)
        assert [g.text for g in results] == [f"echo: p{i}" for i in range(5)]

    def test_persists_across_processes(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.sqlite"
        with GenerationCache(path) as cache:
            asyncio.run(GenerationEngine(CountingBackend(), cache=cache).generate([_request("x")]))
        with GenerationCache(path) as cache:
            backend = CountingBackend()
            asyncio.run(GenerationEngine(backend, cache=cache).generate([_request("x")]))
            assert len(cache) == 1
        assert backend.calls == 0

    @pytest.mark.parametrize(
        ("backend_tag", "request_"),
        [
            ("b", _request("x")),
            ("a", _request("y")),
            ("a", _request("x", sampling=SamplingParams(temperature=0.7))),
            ("a", _request("x", sampling=SamplingParams(max_tokens=10))),
        ],
    )
    def test_key_depends_on_model_prompt_and_sampling(
        self, tmp_path: Path, backend_tag: str, request_: GenerationRequest
    ) -> None:
        with GenerationCache(tmp_path / "cache.sqlite") as cache:
            asyncio.run(
                GenerationEngine(CountingBackend(tag="a"), cache=cache).generate([_request("x")])
            )
            backend = CountingBackend(tag=backend_tag)
            asyncio.run(GenerationEngine(backend, cache=cache).generate([request_]))
        assert backend.calls == 1

    def test_item_id_is_ignored_by_live_backends(self) -> None:
        backend = CountingBackend()
        key_a = cache_key(
            backend.identity(),
            backend.request_identity(_request("x", "a")),
            _request("x", "a").payload(),
        )
        key_b = cache_key(
            backend.identity(),
            backend.request_identity(_request("x", "b")),
            _request("x", "b").payload(),
        )
        assert key_a == key_b

    def test_mock_answers_are_cached_per_item(self, tmp_path: Path) -> None:
        items = _items(4)
        backend = MockBackend(_mock("a"), items)
        with GenerationCache(tmp_path / "cache.sqlite") as cache:
            engine = GenerationEngine(backend, cache=cache)
            asyncio.run(engine.generate([_request("same", "q0"), _request("same", "q1")]))
            asyncio.run(engine.generate([_request("same", "q0"), _request("same", "q1")]))
        assert backend.calls == 2

    def test_backends_that_opt_out_are_not_cached(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path / "preds.jsonl", [{"id": "q0", "output": "x"}])
        with GenerationCache(tmp_path / "cache.sqlite") as cache:
            engine = GenerationEngine(ReplayBackend("r", path), cache=cache)
            asyncio.run(engine.generate([_request("p", "q0")]))
            assert len(cache) == 0

    def test_default_directory_honours_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LFE_CACHE_DIR", str(tmp_path / "explicit"))
        assert default_cache_dir() == tmp_path / "explicit"
        monkeypatch.delenv("LFE_CACHE_DIR")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        assert default_cache_dir() == tmp_path / "xdg" / "llm-finetune-evaluator"


class TestReplay:
    def test_replays_predictions_by_item_id(self, tmp_path: Path) -> None:
        path = write_jsonl(
            tmp_path / "preds.jsonl", [{"id": "a", "output": "x"}, {"id": "b", "output": {"k": 1}}]
        )
        backend = ReplayBackend("recorded", path)
        texts = [
            g.text
            for g in asyncio.run(backend.generate_batch([_request("p", "b"), _request("p", "a")]))
        ]
        assert texts == ['{"k": 1}', "x"]
        with pytest.raises(BackendError, match="no prediction for item 'c'"):
            asyncio.run(backend.generate_batch([_request("p", "c")]))

    def test_custom_field_names(self, tmp_path: Path) -> None:
        path = write_jsonl(tmp_path / "preds.jsonl", [{"uid": "a", "pred": "x"}])
        assert ReplayBackend("r", path, id_field="uid", output_field="pred").predictions == {
            "a": "x"
        }

    @pytest.mark.parametrize(
        ("content", "message"),
        [
            ("{bad", "invalid JSON"),
            ('{"id": "a"}', "expected an object"),
            ('{"id": "a", "output": "x"}\n' * 2, "duplicate id"),
        ],
    )
    def test_rejects_malformed_files(self, tmp_path: Path, content: str, message: str) -> None:
        path = tmp_path / "preds.jsonl"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(DatasetError, match=message):
            ReplayBackend("r", path)


class TestFactory:
    def test_requires_the_api_key_when_one_is_expected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = OpenAIModelConfig(name="judge", backend="openai", model="gpt-4o-mini")
        with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
            create_backend(config)

    def test_local_servers_need_no_key(self) -> None:
        config = OpenAIModelConfig(
            name="base", backend="openai", model="m", base_url="http://localhost:8000/v1"
        )
        backend = create_backend(config)
        assert isinstance(backend, OpenAICompatibleBackend)
        asyncio.run(backend.aclose())

    def test_builds_mock_backends(self) -> None:
        assert isinstance(create_backend(_mock("a"), _items(4)), MockBackend)
        assert isinstance(
            create_backend(MockJudgeConfig(name="j", backend="mock_judge")), MockJudgeBackend
        )


def test_path_fingerprint_changes_when_a_checkpoint_is_rewritten(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"v1")
    first = path_fingerprint(str(adapter))
    weights.write_bytes(b"v2-longer")
    assert first is not None
    assert path_fingerprint(str(adapter)) != first
    assert path_fingerprint("Qwen/Qwen2.5-1.5B-Instruct") is None
