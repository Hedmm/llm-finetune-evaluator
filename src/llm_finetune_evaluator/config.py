"""Experiment configuration, loaded from YAML.

A minimal config::

    name: support-bot
    eval_sets:
      - name: heldout
        path: data/heldout.jsonl
    models:
      - name: base
        backend: openai
        model: Qwen/Qwen2.5-7B-Instruct
        base_url: http://localhost:8000/v1
      - name: finetuned
        backend: openai
        model: support-lora
        base_url: http://localhost:8000/v1
    scorers:
      - type: normalized_match
      - type: refusal

Relative paths are resolved against the directory of the config file. String
values may reference environment variables as ``${NAME}`` or
``${NAME:-default}``. See ``docs/configuration.md`` for every option.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    model_validator,
)

from llm_finetune_evaluator.dataset import PromptTemplate, TaskType
from llm_finetune_evaluator.errors import ConfigError

_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
Name = Annotated[str, Field(pattern=_NAME_PATTERN, max_length=80)]
"""Identifier used in reports and as a directory name."""

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _resolve_path(value: Path, info: ValidationInfo) -> Path:
    value = value.expanduser()
    base_dir = (info.context or {}).get("base_dir")
    if base_dir is not None and not value.is_absolute():
        return Path(base_dir) / value
    return value


ConfigPath = Annotated[Path, AfterValidator(_resolve_path)]
"""A path that is resolved relative to the config file's directory."""


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SamplingParams(_Strict):
    """Decoding parameters sent with every generation request.

    Greedy decoding (``temperature: 0``) is the default because evaluation
    should measure the model, not the luck of one sample.
    """

    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    max_tokens: int = Field(256, ge=1)
    seed: int | None = None
    stop: tuple[str, ...] = ()


class SamplingOverrides(_Strict):
    """Per-model changes to the experiment-wide sampling parameters."""

    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(None, ge=1)
    seed: int | None = None
    stop: tuple[str, ...] | None = None

    def apply(self, base: SamplingParams) -> SamplingParams:
        """Return ``base`` with every field that is set here replaced."""
        return SamplingParams.model_validate(
            {**base.model_dump(), **self.model_dump(exclude_none=True)}
        )


class _ModelConfigBase(_Strict):
    name: Name
    sampling: SamplingOverrides = Field(default_factory=SamplingOverrides)
    max_concurrency: int = Field(4, ge=1, le=256)
    strip_think_tags: bool = True
    cache_tag: str | None = None


class OpenAIModelConfig(_ModelConfigBase):
    """A model behind an OpenAI-compatible HTTP API (vLLM, llama.cpp, ollama, OpenAI)."""

    backend: Literal["openai"]
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str | None = None
    api: Literal["chat", "completions"] = "chat"
    timeout_s: float = Field(120.0, gt=0)
    extra_body: dict[str, Any] = Field(default_factory=dict)

    def resolved_api_key_env(self) -> str | None:
        """Environment variable that holds the API key, if any.

        Defaults to ``OPENAI_API_KEY`` for api.openai.com only, so an OpenAI key
        is never sent to a local server by accident.
        """
        if self.api_key_env is not None:
            return self.api_key_env
        return "OPENAI_API_KEY" if "api.openai.com" in self.base_url else None


class TransformersModelConfig(_ModelConfigBase):
    """A local Hugging Face model, optionally with a PEFT (LoRA) adapter."""

    backend: Literal["transformers"]
    model: str
    adapter: str | None = None
    revision: str | None = None
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    device: str | None = None
    batch_size: int = Field(8, ge=1)
    use_chat_template: bool = True
    trust_remote_code: bool = False


class MockSkill(_Strict):
    """Skill overrides of the mock backend for one tag value."""

    accuracy: float | None = Field(None, ge=0.0, le=1.0)
    partial_rate: float | None = Field(None, ge=0.0, le=1.0)
    refusal_rate: float | None = Field(None, ge=0.0, le=1.0)
    abstain_rate: float | None = Field(None, ge=0.0, le=1.0)


class MockModelConfig(_ModelConfigBase):
    """A deterministic simulated model for tests and offline demos.

    The mock looks up each item in the eval set and answers correctly with
    probability ``accuracy``. Outcomes of different mock models are correlated
    through a shared per-item difficulty (``item_correlation``), as they are
    for real models. Without an explicit ``seed`` the seed is derived from the
    model name, so two mock models never share their private randomness.
    """

    backend: Literal["mock"]
    accuracy: float = Field(0.7, ge=0.0, le=1.0)
    partial_rate: float = Field(0.0, ge=0.0, le=1.0)
    refusal_rate: float = Field(0.02, ge=0.0, le=1.0)
    abstain_rate: float = Field(0.5, ge=0.0, le=1.0)
    by_tag: dict[str, MockSkill] = Field(default_factory=dict)
    item_correlation: float = Field(0.6, ge=0.0, le=1.0)
    seed: int | None = None
    responses: dict[str, str] = Field(default_factory=dict)
    latency_s: float = Field(0.0, ge=0.0)

    @model_validator(mode="after")
    def _check(self) -> MockModelConfig:
        if self.accuracy + self.partial_rate > 1.0:
            raise ValueError("accuracy + partial_rate must not exceed 1")
        for key in self.by_tag:
            if "=" not in key:
                raise ValueError(f"by_tag keys look like 'topic=billing', got {key!r}")
        return self


class MockJudgeConfig(_ModelConfigBase):
    """A deterministic heuristic judge for tests and offline demos.

    It grades by normalised string match and refusal patterns, then flips a
    share ``noise`` of its verdicts so that calibration has something to find.
    """

    backend: Literal["mock_judge"]
    noise: float = Field(0.1, ge=0.0, le=1.0)
    seed: int = 0


class ReplayModelConfig(_ModelConfigBase):
    """Recorded predictions read from a JSONL file instead of a live model."""

    backend: Literal["replay"]
    predictions: ConfigPath
    id_field: str = "id"
    output_field: str = "output"


ModelConfig = Annotated[
    OpenAIModelConfig
    | TransformersModelConfig
    | MockModelConfig
    | MockJudgeConfig
    | ReplayModelConfig,
    Field(discriminator="backend"),
]


class PromptConfig(PromptTemplate):
    """A prompt template whose variables can also come from text files.

    ``variable_files`` maps a variable name to a file; the variable becomes the
    list of the file's non-empty lines, for example a label set shared with a
    ``label_set`` scorer: ``{{ labels | join(", ") }}``.
    """

    variable_files: dict[str, ConfigPath] = Field(default_factory=dict)

    def resolve(self) -> PromptTemplate:
        """Return the template with file variables loaded.

        Raises:
            ConfigError: If a file is missing or a name is defined twice.
        """
        variables = dict(self.variables)
        for name, path in self.variable_files.items():
            if name in variables:
                raise ConfigError(f"prompt variable {name!r} is set both inline and from a file")
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except FileNotFoundError as error:
                raise ConfigError(f"prompt variable file not found: {path}") from error
            variables[name] = [line.strip() for line in lines if line.strip()]
        return PromptTemplate(system=self.system, template=self.template, variables=variables)


class EvalSetConfig(_Strict):
    """An eval set file and the prompt used to present its items."""

    name: Name
    path: ConfigPath
    prompt: PromptConfig = Field(default_factory=PromptConfig)


class _ScorerConfigBase(_Strict):
    name: Name | None = None
    tasks: tuple[TaskType, ...] | None = None
    eval_sets: tuple[str, ...] | None = None

    default_name: ClassVar[str] = ""

    @property
    def resolved_name(self) -> str:
        """The metric name used in reports."""
        return self.name or self.default_name


class ExactMatchConfig(_ScorerConfigBase):
    """Output equals a reference after stripping whitespace."""

    type: Literal["exact_match"]
    case_sensitive: bool = True
    default_name: ClassVar[str] = "exact_match"


class NormalizedMatchConfig(_ScorerConfigBase):
    """Output matches a reference after case, punctuation and article normalisation."""

    type: Literal["normalized_match"]
    mode: Literal["exact", "contains"] = "exact"
    default_name: ClassVar[str] = "normalized_match"


class LabelSetConfig(_ScorerConfigBase):
    """Classification against a closed label set; off-list answers count as wrong."""

    type: Literal["label_set"]
    labels: tuple[str, ...] | None = None
    labels_file: ConfigPath | None = None
    multi_label: bool = False
    default_name: ClassVar[str] = "label_accuracy"

    @model_validator(mode="after")
    def _check(self) -> LabelSetConfig:
        if self.labels is not None and self.labels_file is not None:
            raise ValueError("give either labels or labels_file, not both")
        return self


class JsonFieldsConfig(_ScorerConfigBase):
    """Field-level precision, recall and F1 of a JSON object against the reference."""

    type: Literal["json_fields"]
    numeric_tolerance: float = Field(1e-6, ge=0.0)
    case_sensitive: bool = False
    default_name: ClassVar[str] = "field_f1"


class NumericConfig(_ScorerConfigBase):
    """A number extracted from the output is within tolerance of the reference."""

    type: Literal["numeric"]
    abs_tol: float = Field(0.0, ge=0.0)
    rel_tol: float = Field(1e-9, ge=0.0)
    extract: Literal["first", "last", "only"] = "last"
    default_name: ClassVar[str] = "numeric_match"


class RegexConfig(_ScorerConfigBase):
    """The output matches a regular expression (fixed, or taken from the reference)."""

    type: Literal["regex"]
    pattern: str | None = None
    from_reference: bool = False
    ignore_case: bool = False
    full_match: bool = False
    default_name: ClassVar[str] = "regex_match"

    @model_validator(mode="after")
    def _check(self) -> RegexConfig:
        if (self.pattern is None) == (not self.from_reference):
            raise ValueError("set exactly one of pattern or from_reference: true")
        if self.pattern is not None:
            try:
                re.compile(self.pattern)
            except re.error as error:
                raise ValueError(f"invalid regular expression: {error}") from error
        return self


class RefusalConfig(_ScorerConfigBase):
    """Share of answerable items the model refused (needless refusals)."""

    type: Literal["refusal"]
    extra_patterns: tuple[str, ...] = ()
    default_name: ClassVar[str] = "needless_refusal"


class TrapConfig(_ScorerConfigBase):
    """Trap and refusal_expected items: abstaining is correct, answering is not."""

    type: Literal["trap"]
    extra_patterns: tuple[str, ...] = ()
    default_name: ClassVar[str] = "trap_accuracy"


class JudgeScorerConfig(_ScorerConfigBase):
    """Rubric grading by a separate judge model (see the ``judge`` section)."""

    type: Literal["judge"]
    partial_credit: float = Field(0.5, ge=0.0, le=1.0)
    default_name: ClassVar[str] = "judge_score"


ScorerConfig = Annotated[
    ExactMatchConfig
    | NormalizedMatchConfig
    | LabelSetConfig
    | JsonFieldsConfig
    | NumericConfig
    | RegexConfig
    | RefusalConfig
    | TrapConfig
    | JudgeScorerConfig,
    Field(discriminator="type"),
]


class CalibrationConfig(_Strict):
    """Human-labelled sample used to check the judge."""

    labels: ConfigPath
    kappa_threshold: float = Field(0.6, ge=-1.0, le=1.0)


class JudgeConfig(_Strict):
    """The judge model and how it grades."""

    model: ModelConfig
    sampling: SamplingParams = Field(
        default_factory=lambda: SamplingParams(temperature=0.0, max_tokens=300)
    )
    instructions: str | None = None
    calibration: CalibrationConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def _default_model_name(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("model"), dict):
            data = {**data, "model": {"name": "judge", **data["model"]}}
        return data

    @model_validator(mode="after")
    def _check(self) -> JudgeConfig:
        if isinstance(self.model, (MockModelConfig, ReplayModelConfig)):
            raise ValueError(
                f"backend {self.model.backend!r} cannot act as a judge; "
                "use a real model or 'mock_judge'"
            )
        return self


class StatsConfig(_Strict):
    """Settings for intervals, tests and slicing."""

    alpha: float = Field(0.05, gt=0.0, lt=1.0)
    confidence: float = Field(0.95, gt=0.0, lt=1.0)
    bootstrap_resamples: int = Field(10_000, ge=100)
    permutation_resamples: int = Field(10_000, ge=100)
    seed: int = Field(0, ge=0)
    min_slice_size: int = Field(10, ge=2)
    slice_by: tuple[str, ...] | None = None
    power: float = Field(0.8, ge=0.5, lt=1.0)


class ReportConfig(_Strict):
    """Report contents."""

    title: str | None = None
    primary_metric: str | None = None
    max_examples: int = Field(6, ge=0, le=100)


class CacheConfig(_Strict):
    """On-disk generation cache."""

    enabled: bool = True
    dir: ConfigPath | None = None


class RetryConfig(_Strict):
    """Retries with exponential backoff for transient backend errors."""

    max_attempts: int = Field(5, ge=1)
    initial_delay_s: float = Field(1.0, ge=0.0)
    max_delay_s: float = Field(60.0, ge=0.0)
    multiplier: float = Field(2.0, ge=1.0)


class ExperimentConfig(_Strict):
    """Everything needed to run and report one experiment."""

    name: Name
    description: str | None = None
    output_dir: ConfigPath | None = None
    eval_sets: list[EvalSetConfig] = Field(min_length=1)
    models: list[ModelConfig] = Field(min_length=1)
    baseline: str | None = None
    sampling: SamplingParams = Field(default_factory=SamplingParams)
    scorers: list[ScorerConfig] = Field(min_length=1)
    judge: JudgeConfig | None = None
    stats: StatsConfig = Field(default_factory=StatsConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)

    @model_validator(mode="after")
    def _check(self) -> ExperimentConfig:
        _require_unique("model", [model.name for model in self.models])
        _require_unique("eval set", [eval_set.name for eval_set in self.eval_sets])
        _require_unique("scorer", [scorer.resolved_name for scorer in self.scorers])
        model_names = {model.name for model in self.models}
        if self.baseline is not None and self.baseline not in model_names:
            raise ValueError(
                f"baseline {self.baseline!r} is not one of the models {sorted(model_names)}"
            )
        for model in self.models:
            if isinstance(model, MockJudgeConfig):
                raise ValueError(
                    f"model {model.name!r}: 'mock_judge' can only be used as the judge"
                )
        eval_set_names = {eval_set.name for eval_set in self.eval_sets}
        for scorer in self.scorers:
            unknown = set(scorer.eval_sets or ()) - eval_set_names
            if unknown:
                raise ValueError(
                    f"scorer {scorer.resolved_name!r} refers to unknown eval sets {sorted(unknown)}"
                )
        if self.judge is None and any(isinstance(s, JudgeScorerConfig) for s in self.scorers):
            raise ValueError("a 'judge' scorer needs a 'judge' section with the judge model")
        scorer_names = {scorer.resolved_name for scorer in self.scorers}
        primary = self.report.primary_metric
        if primary is not None and primary not in scorer_names:
            raise ValueError(
                f"report.primary_metric {primary!r} is not one of {sorted(scorer_names)}"
            )
        return self

    @property
    def baseline_name(self) -> str:
        """Name of the baseline model: ``baseline`` if set, else the first model."""
        return self.baseline or self.models[0].name

    def resolved_output_dir(self) -> Path:
        """Directory for this experiment's runs: ``output_dir`` or ``runs/<name>``."""
        return self.output_dir if self.output_dir is not None else Path("runs") / self.name


def _require_unique(kind: str, names: list[str]) -> None:
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{kind} names must be unique; repeated: {duplicates}")


def expand_env(value: Any) -> Any:
    """Replace ``${NAME}`` and ``${NAME:-default}`` in every string of ``value``.

    Raises:
        ConfigError: If a variable without a default is not set.
    """
    if isinstance(value, str):

        def substitute(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise ConfigError(f"environment variable {name} is not set (used as ${{{name}}})")

        return _ENV_PATTERN.sub(substitute, value)
    if isinstance(value, list):
        return [expand_env(element) for element in value]
    if isinstance(value, dict):
        return {key: expand_env(element) for key, element in value.items()}
    return value


def format_validation_error(error: ValidationError, source: str) -> str:
    """Turn a pydantic error into a readable multi-line message."""
    lines = [f"{source} is not a valid config:"]
    for detail in error.errors():
        location = ".".join(str(part) for part in detail["loc"])
        message = detail["msg"].removeprefix("Value error, ")
        lines.append(f"  {location or '(top level)'}: {message}")
    return "\n".join(lines)


def parse_config(
    data: Any, *, base_dir: Path | None = None, source: str = "config"
) -> ExperimentConfig:
    """Validate a config mapping.

    Args:
        data: Parsed YAML.
        base_dir: Directory that relative paths are resolved against.
        source: Name used in error messages.

    Raises:
        ConfigError: If the config is invalid.
    """
    if not isinstance(data, dict):
        raise ConfigError(f"{source} must be a YAML mapping")
    try:
        return ExperimentConfig.model_validate(
            expand_env(data), context={"base_dir": base_dir} if base_dir is not None else None
        )
    except ValidationError as error:
        raise ConfigError(format_validation_error(error, source)) from error


def load_config(path: Path) -> ExperimentConfig:
    """Load and validate a YAML config file.

    Raises:
        ConfigError: If the file is missing, is not valid YAML, or is invalid.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ConfigError(f"config file not found: {path}") from error
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError(f"{path} is not valid YAML: {error}") from error
    return parse_config(data, base_dir=path.parent, source=str(path))
