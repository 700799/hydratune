"""CLI tests via Typer's test runner — no training dependencies required."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

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


def test_export_is_a_stub(tmp_path: Path) -> None:
    config = write_config(tmp_path, "yahma/alpaca-cleaned")
    result = runner.invoke(app, ["export", "--config", str(config)])
    assert result.exit_code == 2
    assert "not implemented" in result.output


@pytest.mark.parametrize("missing", ["missing.yaml"])
def test_missing_config_file(missing: str) -> None:
    result = runner.invoke(app, ["validate", "--config", missing])
    assert result.exit_code != 0
