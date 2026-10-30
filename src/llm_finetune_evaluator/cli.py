"""Command-line interface: ``lfe``.

Commands:

* ``lfe run``: generate and score with every model of a config.
* ``lfe compare``: paired comparison of two runs in the terminal.
* ``lfe report``: Markdown, HTML and JSON reports for an experiment directory.
* ``lfe judge-calibrate``: check the LLM judge against human labels.
* ``lfe mde``: eval set sizing (minimum detectable effect and sample size).
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, TypeVar

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from llm_finetune_evaluator import __version__
from llm_finetune_evaluator.analysis import (
    AnalysisSettings,
    Comparison,
    ExperimentAnalysis,
    analyze,
)
from llm_finetune_evaluator.backends import create_backend
from llm_finetune_evaluator.backends.cache import GenerationCache
from llm_finetune_evaluator.backends.engine import GenerationEngine, ProgressCallback, RetryPolicy
from llm_finetune_evaluator.calibration import (
    CALIBRATION_FILE_NAME,
    CalibrationResult,
    calibrate_judge,
    labeling_template,
    load_calibration,
    load_human_labels,
    save_calibration,
)
from llm_finetune_evaluator.config import (
    ExperimentConfig,
    JudgeScorerConfig,
    ModelConfig,
    OpenAIModelConfig,
    load_config,
)
from llm_finetune_evaluator.errors import ConfigError, LfeError
from llm_finetune_evaluator.report import REPORT_FORMATS, write_reports
from llm_finetune_evaluator.report import formatting as fmt
from llm_finetune_evaluator.runner import (
    judge_family_warnings,
    load_eval_sets,
    prepare_items,
    run_experiment,
)
from llm_finetune_evaluator.scorers import JudgeScorer, build_scorer
from llm_finetune_evaluator.stats import (
    minimum_detectable_effect,
    minimum_detectable_effect_continuous,
    required_sample_size,
)
from llm_finetune_evaluator.storage import RunData, discover_runs, load_experiment, load_run

console = Console()
error_console = Console(stderr=True)

app = typer.Typer(
    name="lfe",
    help=(
        "Is your fine-tuned LLM actually better than the base model, and where does it get "
        "worse? Run every model on the same eval set, score the answers, and compare them "
        "with paired statistics."
    ),
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="markdown",
    pretty_exceptions_enable=False,
)

F = TypeVar("F", bound=Callable[..., Any])


def _friendly_errors(function: F) -> F:
    """Print expected errors as one red line instead of a traceback."""

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except LfeError as error:
            error_console.print(f"[bold red]error:[/bold red] {error}", highlight=False)
            raise typer.Exit(code=1) from error
        except KeyboardInterrupt as error:
            error_console.print(
                "[yellow]Interrupted.[/yellow] Completed generations are cached; "
                "run the same command again to resume."
            )
            raise typer.Exit(code=130) from error

    return wrapper  # type: ignore[return-value]


def _version(value: bool) -> None:
    if value:
        console.print(f"llm-finetune-evaluator {__version__}")
        raise typer.Exit()


@app.callback()
def main_options(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show retries and other details.")
    ] = False,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show the version."),
    ] = False,
) -> None:
    """Paired-statistics evaluation of fine-tuned LLMs."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=error_console, show_path=False, show_time=False)],
        force=True,
    )


@contextmanager
def _progress(enabled: bool) -> Iterator[Callable[[str, int], ProgressCallback] | None]:
    if not enabled:
        yield None
        return
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=error_console,
        transient=False,
    ) as progress:

        def factory(description: str, total: int) -> ProgressCallback:
            task = progress.add_task(description, total=total)
            return lambda count: progress.advance(task, count)

        yield factory


def _estimate_cell(analysis: ExperimentAnalysis, metric: str, model: str) -> str:
    return fmt.estimate(analysis.headline[metric][model].estimate)


def _verdict_markup(comparison: Comparison) -> str:
    if comparison.outcome == "worse":
        return "[red]▼ worse[/red]"
    if comparison.outcome == "better":
        return "[green]▲ better[/green]"
    return "[dim]not significant[/dim]"


def _print_analysis(analysis: ExperimentAnalysis) -> None:
    headline = Table(title="Headline metrics", title_justify="left", show_lines=False)
    headline.add_column("Metric")
    for model in analysis.models:
        label = f"{model} (baseline)" if model == analysis.baseline else model
        headline.add_column(label, justify="right")
    for metric in analysis.metrics:
        name = metric.name + ("" if metric.higher_is_better else " (lower is better)")
        if analysis.judge_unreliable(metric):
            name += " [yellow]⚠ unreliable judge[/yellow]"
        headline.add_row(name, *(_estimate_cell(analysis, metric.name, m) for m in analysis.models))
    console.print(headline)
    comparisons = [c for c in analysis.pairwise if c.baseline == analysis.baseline]
    if comparisons:
        several = len(analysis.candidates) > 1
        table = Table(title=f"Against {analysis.baseline} (paired tests)", title_justify="left")
        table.add_column("Metric")
        if several:
            table.add_column("Candidate")
        table.add_column("Difference (CI)", justify="right")
        table.add_column("p (Holm)", justify="right")
        table.add_column("MDE", justify="right")
        table.add_column("Verdict")
        for c in comparisons:
            cells = [c.metric] + ([c.candidate] if several else [])
            cells += [
                f"{fmt.points(c.difference)} ({fmt.interval(c.ci_low, c.ci_high, signed=True)})",
                fmt.p_value(c.p_adjusted),
                fmt.mde(c.mde),
                _verdict_markup(c),
            ]
            table.add_row(*cells)
        console.print(table)
    if analysis.regressions:
        console.print(f"[bold red]{len(analysis.regressions)} regression(s):[/bold red]")
        for c in analysis.regressions:
            console.print(
                f"  [red]▼[/red] {c.candidate} on {c.metric}, {c.slice_label}: "
                f"{fmt.points(c.difference)} ({fmt.interval(c.ci_low, c.ci_high, signed=True)}), "
                f"n = {c.n}, Holm p = {fmt.p_value(c.p_adjusted)}",
                highlight=False,
            )
    elif analysis.candidates:
        console.print(
            "[green]No regressions[/green] after Holm correction, overall or in any slice."
        )
    for warning in analysis.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}", highlight=False)


def _settings(
    experiment: dict[str, Any] | None,
    alpha: float | None,
    resamples: int | None,
    seed: int | None,
    min_slice_size: int | None = None,
) -> AnalysisSettings:
    return AnalysisSettings.from_experiment(
        experiment,
        alpha=alpha,
        bootstrap_resamples=resamples,
        permutation_resamples=resamples,
        seed=seed,
        min_slice_size=min_slice_size,
    )


RUN_EPILOG = """
**Examples**

* `lfe run examples/mock_quickstart/config.yaml` runs the offline demo.

* `lfe run config.yaml --dry-run` checks the config, eval sets and prompts without calling a model.

* `lfe run config.yaml --model finetuned` re-runs one model; other runs are kept.

* `lfe run config.yaml --report` also writes report.md and report.html.

Generations are cached on disk (`$LFE_CACHE_DIR`, default `~/.cache/llm-finetune-evaluator`),
so re-running or re-scoring costs nothing.
"""


def _missing_keys(config: ExperimentConfig) -> list[str]:
    models: list[ModelConfig] = list(config.models)
    if config.judge is not None:
        models.append(config.judge.model)
    missing = []
    for model in models:
        if isinstance(model, OpenAIModelConfig):
            variable = model.resolved_api_key_env()
            if variable and not os.environ.get(variable):
                missing.append(f"{model.name} needs ${variable}")
    return missing


def _dry_run(config: ExperimentConfig, config_path: Path) -> None:
    eval_sets = load_eval_sets(config)
    prepared = prepare_items(config, eval_sets)
    console.print(f"[bold]{config_path}[/bold] is valid.", highlight=False)
    sets = Table(title="Eval sets", title_justify="left")
    for column in ("Name", "File", "Items", "Tasks"):
        sets.add_column(column)
    for eval_set in eval_sets:
        tasks: dict[str, int] = {}
        for item in eval_set.items:
            tasks[item.task.value] = tasks.get(item.task.value, 0) + 1
        sets.add_row(
            eval_set.name,
            str(eval_set.path),
            str(len(eval_set.items)),
            ", ".join(f"{task} {count}" for task, count in sorted(tasks.items())),
        )
    console.print(sets)
    models = Table(title="Models", title_justify="left")
    for column in ("Name", "Backend", "Model", "Sampling"):
        models.add_column(column)
    for model in config.models:
        sampling = model.sampling.apply(config.sampling)
        baseline = " (baseline)" if model.name == config.baseline_name else ""
        models.add_row(
            model.name + baseline,
            model.backend,
            str(getattr(model, "model", getattr(model, "predictions", "simulated"))),
            f"temperature={sampling.temperature}, max_tokens={sampling.max_tokens}",
        )
    console.print(models)
    scorers = Table(title="Scorers", title_justify="left")
    for column in ("Metric", "Type", "Items"):
        scorers.add_column(column)
    for scorer_config in config.scorers:
        if isinstance(scorer_config, JudgeScorerConfig):
            tasks_in_scope = set(scorer_config.tasks or JudgeScorer.default_tasks)
            count = sum(
                entry.item.task in tasks_in_scope
                and (scorer_config.eval_sets is None or entry.eval_set in scorer_config.eval_sets)
                for entry in prepared
            )
        else:
            scorer = build_scorer(scorer_config, config, eval_sets)
            count = sum(scorer.applies_to(entry.item, entry.eval_set) for entry in prepared)
        scorers.add_row(scorer_config.resolved_name, scorer_config.type, str(count))
    console.print(scorers)
    if config.judge is not None:
        judge_model = getattr(config.judge.model, "model", "simulated")
        console.print(f"Judge: {config.judge.model.backend} ({judge_model})", highlight=False)
    for warning in judge_family_warnings(config):
        console.print(f"[yellow]Warning:[/yellow] {warning}", highlight=False)
    for missing in _missing_keys(config):
        console.print(f"[yellow]Warning:[/yellow] {missing} (not set)", highlight=False)
    for eval_set in eval_sets:
        first = next(entry for entry in prepared if entry.eval_set == eval_set.name)
        console.print(f"\n[bold]First prompt of {eval_set.name}[/bold] (item {first.item.id}):")
        for message in first.messages:
            console.print(
                f"{message.role}: {fmt.truncate(message.content, 800)}",
                highlight=False,
                markup=False,
            )


@app.command(epilog=RUN_EPILOG)
@_friendly_errors
def run(
    config: Annotated[
        Path, typer.Argument(help="Experiment config (YAML).", metavar="CONFIG", show_default=False)
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir", "-o", help="Where to write runs (default: runs/EXPERIMENT_NAME)."
        ),
    ] = None,
    model: Annotated[
        list[str] | None,
        typer.Option("--model", "-m", help="Run only this model; repeat for several."),
    ] = None,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Ignore and do not update the generation cache.")
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate config, eval sets and prompts; call no model."),
    ] = False,
    report: Annotated[
        bool, typer.Option("--report", help="Write the Markdown and HTML reports afterwards.")
    ] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="No progress bars.")] = False,
) -> None:
    """Generate and score with every model in the config, then print a summary."""
    experiment_config = load_config(config)
    if dry_run:
        _dry_run(experiment_config, config)
        return
    with _progress(not quiet) as progress:
        result = run_experiment(
            experiment_config,
            output_dir=output_dir,
            models=model,
            use_cache=not no_cache,
            config_path=config,
            progress=progress,
        )
    console.print(f"Runs written to [bold]{result.output_dir}[/bold]", highlight=False)
    experiment = load_experiment(result.output_dir)
    runs = [load_run(path) for path in discover_runs(result.output_dir)]
    analysis = analyze(
        runs,
        settings=_settings(experiment, None, None, None),
        calibration=load_calibration(result.output_dir / CALIBRATION_FILE_NAME),
        experiment=experiment,
    )
    _print_analysis(analysis)
    if report:
        written = write_reports(analysis, result.output_dir, ("md", "html", "json"))
        for path in written.values():
            console.print(f"Wrote {path}", highlight=False)
    else:
        console.print(f"\nNext: [bold]lfe report {result.output_dir}[/bold]", highlight=False)


COMPARE_EPILOG = """
**Examples**

* `lfe compare runs/support/base runs/support/finetuned`

* `lfe compare runs/a/base runs/b/base --out reports/a-vs-b` also writes the reports.

* `lfe compare runs/x/base runs/x/lora --fail-on-regression` exits with code 2 on a
  regression, for CI gates.

Both runs must be scored on the same eval set content (checked by hash).
"""


def _relabel(runs: list[RunData]) -> list[RunData]:
    """Give runs of the same model name distinct labels (``<experiment>/<model>``)."""
    names = [run.model for run in runs]
    if len(set(names)) == len(names):
        return runs
    return [
        replace(run, meta={**run.meta, "model": f"{run.path.parent.name}/{run.model}"})
        for run in runs
    ]


@app.command(epilog=COMPARE_EPILOG)
@_friendly_errors
def compare(
    baseline_run: Annotated[
        Path,
        typer.Argument(
            help="Run directory of the baseline.", metavar="BASELINE_RUN", show_default=False
        ),
    ],
    candidate_run: Annotated[
        Path,
        typer.Argument(
            help="Run directory of the candidate.", metavar="CANDIDATE_RUN", show_default=False
        ),
    ],
    out: Annotated[Path | None, typer.Option("--out", help="Also write the reports here.")] = None,
    alpha: Annotated[float | None, typer.Option(help="Significance level (default 0.05).")] = None,
    resamples: Annotated[
        int | None, typer.Option(help="Bootstrap and permutation resamples.")
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Seed for resampling.")] = None,
    fail_on_regression: Annotated[
        bool,
        typer.Option("--fail-on-regression", help="Exit with code 2 if there is a regression."),
    ] = False,
) -> None:
    """Compare two runs item by item and list the regressions."""
    runs = _relabel([load_run(baseline_run), load_run(candidate_run)])
    experiment = load_experiment(baseline_run.parent)
    calibration = load_calibration(baseline_run.parent / CALIBRATION_FILE_NAME)
    analysis = analyze(
        runs,
        baseline=runs[0].model,
        settings=_settings(experiment, alpha, resamples, seed),
        calibration=calibration,
        experiment=None,
        title=f"Comparison: {runs[1].model} vs {runs[0].model}",
    )
    _print_analysis(analysis)
    if out is not None:
        for path in write_reports(analysis, out).values():
            console.print(f"Wrote {path}", highlight=False)
    if fail_on_regression and analysis.regressions:
        raise typer.Exit(code=2)


REPORT_EPILOG = """
**Examples**

* `lfe report runs/mock_quickstart` writes report.md, report.html and analysis.json there.

* `lfe report runs/support --baseline base --out site/` chooses the baseline and output folder.

* `lfe report runs/support --format html --alpha 0.01`

* `lfe report runs/support --fail-on-regression` exits with code 2 on a regression (for CI).

The report is self-contained: one HTML file with inline CSS and SVG, no network access.
"""


@app.command(epilog=REPORT_EPILOG)
@_friendly_errors
def report(
    runs_dir: Annotated[
        Path,
        typer.Argument(
            help="Experiment directory (or a parent of one).",
            metavar="RUNS_DIR",
            show_default=False,
        ),
    ],
    baseline: Annotated[
        str | None, typer.Option(help="Baseline model (default: from the config).")
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Output folder (default: RUNS_DIR).")
    ] = None,
    report_format: Annotated[
        list[str] | None,
        typer.Option("--format", "-f", help="md, html or json; repeat for several (default: all)."),
    ] = None,
    title: Annotated[str | None, typer.Option(help="Report title.")] = None,
    alpha: Annotated[float | None, typer.Option(help="Significance level (default 0.05).")] = None,
    resamples: Annotated[
        int | None, typer.Option(help="Bootstrap and permutation resamples.")
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Seed for resampling.")] = None,
    min_slice_size: Annotated[
        int | None, typer.Option(help="Smallest slice to report and test.")
    ] = None,
    fail_on_regression: Annotated[
        bool,
        typer.Option("--fail-on-regression", help="Exit with code 2 if there is a regression."),
    ] = False,
) -> None:
    """Write Markdown, HTML and JSON reports for the runs of an experiment."""
    formats = report_format or list(REPORT_FORMATS)
    unknown = sorted(set(formats) - set(REPORT_FORMATS))
    if unknown:
        raise ConfigError(f"unknown report format(s) {unknown}; choose from {list(REPORT_FORMATS)}")
    paths = discover_runs(runs_dir)
    experiment_dir = paths[0].parent
    experiment = load_experiment(experiment_dir)
    runs = [load_run(path) for path in paths]
    analysis = analyze(
        runs,
        baseline=baseline,
        settings=_settings(experiment, alpha, resamples, seed, min_slice_size),
        calibration=load_calibration(experiment_dir / CALIBRATION_FILE_NAME),
        experiment=experiment,
        title=title,
    )
    _print_analysis(analysis)
    for path in write_reports(analysis, out or runs_dir, formats).values():
        console.print(f"Wrote {path}", highlight=False)
    if fail_on_regression and analysis.regressions:
        raise typer.Exit(code=2)


CALIBRATE_EPILOG = """
**Examples**

* `lfe judge-calibrate examples/mock_quickstart/config.yaml` uses the labels file named
  in the config.

* `lfe judge-calibrate config.yaml --labels human_labels.jsonl --threshold 0.7`

* `lfe judge-calibrate config.yaml --export-template to_label.jsonl --size 60` samples outputs
  from the runs for people to label (blind: no model names).

Labels are JSONL lines like `{"id": "faq-17", "output": "...", "label": "correct"}` with one of
correct, partially_correct, incorrect, refused, made_up. The result is saved as calibration.json
in the experiment directory, and `lfe report` marks judge metrics as unreliable when kappa is
below the threshold.
"""


def _print_calibration(result: CalibrationResult) -> None:
    console.print(
        f"Judge [bold]{result.judge}[/bold] on {result.n} human-labelled outputs: "
        f"agreement {fmt.percent(result.agreement)}, Cohen's kappa {fmt.kappa(result.kappa)} "
        f"({result.confidence:.0%} CI {fmt.kappa(result.kappa_low)} to "
        f"{fmt.kappa(result.kappa_high)}), "
        f"threshold {result.threshold:.2f}.",
        highlight=False,
    )
    used = [
        (index, category)
        for index, category in enumerate(result.categories)
        if sum(result.confusion[index]) or any(row[index] for row in result.confusion)
    ]
    table = Table(title="Confusion matrix (rows: human, columns: judge)", title_justify="left")
    table.add_column("human \\ judge")
    for _, category in used:
        table.add_column(category, justify="right")
    for i, category in used:
        table.add_row(category, *(str(result.confusion[i][j]) for j, _ in used))
    console.print(table)
    if result.warning:
        error_console.print(f"[bold red]WARNING:[/bold red] {result.warning}", highlight=False)
    else:
        console.print("[green]The judge agrees well enough with the human labels.[/green]")


@app.command("judge-calibrate", epilog=CALIBRATE_EPILOG)
@_friendly_errors
def judge_calibrate(
    config: Annotated[
        Path,
        typer.Argument(
            help="Experiment config with a 'judge' section.", metavar="CONFIG", show_default=False
        ),
    ],
    labels: Annotated[
        Path | None, typer.Option(help="Human labels (default: judge.calibration.labels).")
    ] = None,
    threshold: Annotated[
        float | None, typer.Option(help="Minimum kappa for a reliable judge (default 0.6).")
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Where to save calibration.json.")
    ] = None,
    export_template: Annotated[
        Path | None,
        typer.Option("--export-template", help="Write a labelling template from the runs instead."),
    ] = None,
    runs_dir: Annotated[
        Path | None, typer.Option(help="Runs to sample for the template (default: the config's).")
    ] = None,
    size: Annotated[int, typer.Option(help="Outputs to sample for the template.")] = 50,
    seed: Annotated[int, typer.Option(help="Seed for sampling and the kappa interval.")] = 0,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Do not use the generation cache.")
    ] = False,
    strict: Annotated[
        bool, typer.Option("--strict", help="Exit with code 2 when the judge is unreliable.")
    ] = False,
) -> None:
    """Check the LLM judge against human labels: agreement and Cohen's kappa."""
    experiment_config = load_config(config)
    if experiment_config.judge is None:
        raise ConfigError(f"{config} has no 'judge' section")
    output_dir = experiment_config.resolved_output_dir()
    eval_sets = load_eval_sets(experiment_config)
    items = {item.id: item for eval_set in eval_sets for item in eval_set.items}
    if export_template is not None:
        source = runs_dir or output_dir
        outputs = [
            (items[record.id], record.output)
            for path in discover_runs(source)
            for record in load_run(path).records
            if record.id in items
        ]
        rows = labeling_template(outputs, size, seed)
        export_template.parent.mkdir(parents=True, exist_ok=True)
        export_template.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        console.print(
            f"Wrote {len(rows)} outputs to label to {export_template}. Fill in each 'label' with "
            "correct, partially_correct, incorrect, refused or made_up, then run "
            f"`lfe judge-calibrate {config} --labels {export_template}`.",
            highlight=False,
        )
        return
    calibration_config = experiment_config.judge.calibration
    labels_path = labels or (calibration_config.labels if calibration_config else None)
    if labels_path is None:
        raise ConfigError(
            "no human labels: pass --labels or set judge.calibration.labels in the config"
        )
    kappa_threshold = (
        threshold
        if threshold is not None
        else (calibration_config.kappa_threshold if calibration_config else 0.6)
    )
    cache = (
        None
        if no_cache or not experiment_config.cache.enabled
        else GenerationCache.open_default(experiment_config.cache.dir)
    )
    backend = create_backend(experiment_config.judge.model)
    engine = GenerationEngine(
        backend, cache=cache, retry=RetryPolicy.from_config(experiment_config.retry)
    )
    scorer = JudgeScorer(
        "judge_score",
        engine=engine,
        sampling=experiment_config.judge.sampling,
        instructions=experiment_config.judge.instructions,
    )

    async def calibrate() -> CalibrationResult:
        try:
            return await calibrate_judge(
                scorer,
                load_human_labels(labels_path),
                items,
                threshold=kappa_threshold,
                seed=seed,
                labels_file=labels_path,
            )
        finally:
            await backend.aclose()

    try:
        result = asyncio.run(calibrate())
    finally:
        if cache is not None:
            cache.close()
    _print_calibration(result)
    destination = out or output_dir / CALIBRATION_FILE_NAME
    save_calibration(result, destination)
    console.print(f"Saved {destination}", highlight=False)
    if strict and not result.reliable:
        raise typer.Exit(code=2)


MDE_EPILOG = """
**Examples**

* `lfe mde --n 300 --discordance 0.15`: smallest detectable accuracy difference with 300 items.

* `lfe mde --effect 0.03 --discordance 0.15`: items needed to detect 3 points.

* `lfe mde --discordance 0.15`: a table for common eval set sizes.

* `lfe mde --n 300 --sd-difference 0.35`: for continuous scores such as judge grades or F1.

The discordance is the share of items on which the two models disagree (one right, one wrong);
`lfe compare` and the reports show it for real runs. See docs/sizing-eval-sets.md.
"""


@app.command(epilog=MDE_EPILOG)
@_friendly_errors
def mde(
    n: Annotated[int | None, typer.Option("--n", help="Number of paired items.")] = None,
    discordance: Annotated[
        float | None, typer.Option(help="Share of items where the models disagree (0 to 1).")
    ] = None,
    effect: Annotated[
        float | None, typer.Option(help="Accuracy difference to detect, e.g. 0.03 for 3 points.")
    ] = None,
    sd_difference: Annotated[
        float | None, typer.Option(help="SD of per-item score differences (continuous metrics).")
    ] = None,
    alpha: Annotated[float, typer.Option(help="Two-sided significance level.")] = 0.05,
    power: Annotated[float, typer.Option(help="Probability of detecting the effect.")] = 0.8,
) -> None:
    """Size an eval set: minimum detectable effect or required number of items."""
    if sd_difference is not None:
        if n is None:
            raise ConfigError("--sd-difference needs --n")
        value = minimum_detectable_effect_continuous(n, sd_difference, alpha=alpha, power=power)
        console.print(
            f"With {n} items and a per-item SD of differences of {sd_difference:g}, the minimum "
            f"detectable difference is {fmt.mde(value)} (alpha = {alpha}, power = {power:.0%}).",
            highlight=False,
        )
        return
    if discordance is None:
        raise ConfigError("pass --discordance (or --sd-difference for continuous scores)")
    try:
        if effect is not None:
            needed = required_sample_size(effect, discordance, alpha=alpha, power=power)
            console.print(
                f"To detect {fmt.points(effect, signed=False)} with discordance "
                f"{discordance:g} you need "
                f"{needed:,} paired items (alpha = {alpha}, power = {power:.0%}).",
                highlight=False,
            )
            return
        sizes: Sequence[int] = [n] if n is not None else [100, 200, 300, 500, 1000, 2000, 5000]
        table = Table(
            title=(
                f"Minimum detectable difference (discordance {discordance:g}, "
                f"alpha {alpha}, power {power:.0%})"
            ),
            title_justify="left",
        )
        table.add_column("Items", justify="right")
        table.add_column("MDE", justify="right")
        for size in sizes:
            detectable = minimum_detectable_effect(size, discordance, alpha=alpha, power=power)
            table.add_row(
                f"{size:,}",
                fmt.mde(detectable) if detectable is not None else "not detectable",
            )
        console.print(table)
    except ValueError as error:
        raise ConfigError(str(error)) from error


def main() -> None:
    """Entry point of the ``lfe`` console script."""
    app()


if __name__ == "__main__":
    main()
