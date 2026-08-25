"""HydraTune command-line interface.

Commands:
    hydratune train    --config config.yaml   Run a fine-tuning job.
    hydratune validate --config config.yaml   Check config + dataset without training.
    hydratune export   --config config.yaml   (planned) Merge adapters / export GGUF.

Heavy training dependencies (torch, TRL, PEFT) are imported only inside
``train``, so ``validate`` works on machines with just the base install.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from hydratune import __version__
from hydratune.config import HydraTuneConfig, load_config
from hydratune.data.formatting import inspect_local_dataset
from hydratune.utils.errors import HydraTuneError

app = typer.Typer(
    name="hydratune",
    help="YAML-configurable LLM fine-tuning harness.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
error_console = Console(stderr=True, style="bold red")

ConfigOption = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        help="Path to the YAML configuration file.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
]


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"hydratune {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True),
    ] = False,
) -> None:
    """YAML-configurable LLM fine-tuning harness."""


def _load_or_exit(config_path: Path) -> HydraTuneConfig:
    try:
        return load_config(config_path)
    except HydraTuneError as exc:
        error_console.print(str(exc))
        raise typer.Exit(code=1) from exc


def _summarize(config: HydraTuneConfig) -> Table:
    table = Table(title="HydraTune run summary", show_header=False, min_width=60)
    table.add_row("Base model", config.base_model.name)
    table.add_row("Dataset", f"{config.dataset.path} ({config.dataset.format})")
    if config.dataset.chat_template is not None:
        table.add_row("Chat template", config.dataset.chat_template)
    if config.peft is not None:
        table.add_row(
            "Adapter",
            f"{config.peft.adapter} (r={config.peft.r}, alpha={config.peft.lora_alpha})",
        )
    else:
        table.add_row("Adapter", "none (full fine-tune)")
    table.add_row("Precision", config.torch_dtype)
    table.add_row(
        "Batch",
        f"{config.training.batch_size} x {config.training.gradient_accumulation_steps} "
        f"accumulation = {config.training.effective_batch_size} effective",
    )
    table.add_row("Sequence length", str(config.training.max_seq_length))
    table.add_row("Output dir", str(config.training.output_dir))
    return table


@app.command()
def train(config: ConfigOption) -> None:
    """Fine-tune a model as described by the config file."""
    run_config = _load_or_exit(config)
    console.print(_summarize(run_config))

    try:
        # Imported here so the base install works without torch/TRL.
        from hydratune.training.trainer import run_training
    except ImportError as exc:
        error_console.print(
            f"Training dependencies are not installed ({exc}).\n"
            'Install them with: pip install "hydratune[train]"'
        )
        raise typer.Exit(code=1) from exc

    try:
        output_dir = run_training(run_config)
    except HydraTuneError as exc:
        error_console.print(str(exc))
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Training complete.[/green] Artifacts saved to {output_dir}")


@app.command()
def validate(
    config: ConfigOption,
    check_dataset: Annotated[
        bool,
        typer.Option(
            "--check-dataset/--no-check-dataset",
            help="Also sample local dataset files and verify records match the declared format.",
        ),
    ] = True,
) -> None:
    """Validate the config file (and optionally the dataset) without training."""
    run_config = _load_or_exit(config)
    console.print("[green]Config OK[/green] — schema validation passed.")
    console.print(_summarize(run_config))

    if not check_dataset:
        return
    if not run_config.dataset.is_local_file:
        console.print(
            f"Dataset {run_config.dataset.path!r} looks like a Hub id; "
            "content checks run at training time."
        )
        return

    problems = inspect_local_dataset(run_config.dataset)
    if problems:
        error_console.print("Dataset check failed:")
        for problem in problems:
            error_console.print(f"  - {problem}")
        raise typer.Exit(code=1)
    console.print("[green]Dataset OK[/green] — sampled records match the declared format.")


@app.command()
def export(config: ConfigOption) -> None:
    """(Planned) Merge adapters into the base model and export to GGUF."""
    _load_or_exit(config)
    error_console.print(
        "hydratune export is not implemented yet; adapter merging and GGUF "
        "conversion are tracked for an upcoming release."
    )
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
