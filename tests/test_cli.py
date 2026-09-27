"""CLI tests via Typer's test runner — no training dependencies required."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hydratune import cli
from hydratune import export as export_pkg
from hydratune.cli import app

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parent.parent


def write_config(tmp_path: Path, dataset_path: str, format: str = "alpaca") -> Path:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
base_model:
  name: meta-llama/Meta-Llama-3-8B
dataset:
  path: {dataset_path}
  format: {format}
peft:
  adapter: qlora
""",
        encoding="utf-8",
    )
    return config


def test_example_config_validates() -> None:
    result = runner.invoke(
        app,
        ["validate", "--config", str(REPO_ROOT / "example_config.yaml")],
    )
    assert result.exit_code == 0, result.output
    assert "Config OK" in result.output


def test_validate_checks_local_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    records = [
        {"instruction": "Say hi", "input": "", "output": "Hi!"},
        {"instruction": "Add 2+2", "input": "", "output": "4"},
    ]
    dataset.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )
    config = write_config(tmp_path, str(dataset))

    result = runner.invoke(app, ["validate", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "Dataset OK" in result.output


def test_validate_flags_format_mismatch(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}))
    config = write_config(tmp_path, str(dataset), format="alpaca")

    result = runner.invoke(app, ["validate", "--config", str(config)])
    assert result.exit_code == 1
    assert "Dataset check failed" in result.output


def test_validate_rejects_bad_config(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("base_model:\n  name: some/model\n", encoding="utf-8")

    result = runner.invoke(app, ["validate", "--config", str(config)])
    assert result.exit_code == 1
    assert "dataset" in result.output


@pytest.mark.parametrize("command", ["train", "export"])
def test_missing_training_extras_is_reported_cleanly(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A base install must get an install hint, never a traceback.

    The heavy imports live inside functions, so importing trainer/merge
    succeeds without the extra and the real ImportError would otherwise
    surface mid-run as an unhandled ModuleNotFoundError.
    """
    monkeypatch.setattr(cli, "_missing_training_modules", lambda: ["torch"])
    config = write_config(tmp_path, "yahma/alpaca-cleaned")

    result = runner.invoke(app, [command, "--config", str(config)])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Training dependencies are not installed" in result.output
    assert "hydratune[train]" in result.output


def test_export_reports_missing_adapter(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    config = write_config(tmp_path, "yahma/alpaca-cleaned")
    result = runner.invoke(
        app, ["export", "--config", str(config), "--adapter", str(tmp_path / "nope")]
    )
    assert result.exit_code == 1
    assert "Adapter directory not found" in result.output


def test_export_output_flag_is_wired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--adapter/--output reach merge_adapter (asserted functionally, not via --help)."""
    seen: dict[str, Path | None] = {}

    def fake_merge(config: object, adapter_dir: Path | None, output_dir: Path | None) -> Path:
        seen["adapter"], seen["output"] = adapter_dir, output_dir
        return Path("/merged")

    monkeypatch.setattr(cli, "_missing_training_modules", list)
    monkeypatch.setattr(export_pkg, "merge_adapter", fake_merge)

    config = write_config(tmp_path, "yahma/alpaca-cleaned")
    result = runner.invoke(
        app,
        ["export", "--config", str(config), "--adapter", str(tmp_path), "--output", "/out"],
    )
    assert result.exit_code == 0, result.output
    assert seen == {"adapter": tmp_path, "output": Path("/out")}


@pytest.mark.parametrize("missing", ["missing.yaml"])
def test_missing_config_file(missing: str) -> None:
    result = runner.invoke(app, ["validate", "--config", missing])
    assert result.exit_code != 0


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "hydratune" in result.output


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "train" in result.output
    assert "validate" in result.output


def test_validate_skips_dataset_check_when_disabled(tmp_path: Path) -> None:
    config = write_config(tmp_path, str(tmp_path / "does_not_exist.jsonl"))
    result = runner.invoke(app, ["validate", "--config", str(config), "--no-check-dataset"])
    assert result.exit_code == 0, result.output
    assert "Dataset" not in result.output.split("└")[-1]


def test_validate_openai_format_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "chat.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            }
        ),
        encoding="utf-8",
    )
    config = write_config(tmp_path, str(dataset), format="openai")
    result = runner.invoke(app, ["validate", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "Dataset OK" in result.output


def test_validate_reports_empty_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("", encoding="utf-8")
    config = write_config(tmp_path, str(dataset))
    result = runner.invoke(app, ["validate", "--config", str(config)])
    assert result.exit_code == 1
    assert "no records" in result.output
