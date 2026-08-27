"""Tests for YAML loading and validation-error reporting."""

from __future__ import annotations

from pathlib import Path

import pytest

from hydratune.config import ConfigError, load_config

VALID_MINIMAL = """
base_model:
  name: some/model
dataset:
  path: some/dataset
"""


def test_valid_config_loads(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(VALID_MINIMAL, encoding="utf-8")
    config = load_config(config_file)
    assert config.base_model.name == "some/model"


def test_accepts_str_path(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(VALID_MINIMAL, encoding="utf-8")
    assert load_config(str(config_file)).dataset.path == "some/dataset"


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_invalid_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("base_model: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(config_file)


def test_empty_file(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_config(config_file)


def test_non_mapping_top_level(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(config_file)


def test_undecodable_file(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(ConfigError, match="could not be read|not valid YAML"):
        load_config(config_file)


def test_validation_errors_report_field_paths(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
base_model:
  name: some/model
dataset:
  path: some/dataset
training:
  learning_rate: -1
peft:
  adapter: dpo
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError) as excinfo:
        load_config(config_file)
    message = str(excinfo.value)
    assert "training.learning_rate" in message
    assert "peft.adapter" in message
    assert "2 error(s)" in message
