"""Tests for dataset format conversion and the lightweight dataset inspector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hydratune.config.schema import DatasetConfig
from hydratune.data.formatting import inspect_local_dataset, to_messages
from hydratune.utils.errors import DatasetError

# --- to_messages ----------------------------------------------------------


def test_alpaca_without_input() -> None:
    record = {"instruction": "Say hi", "input": "", "output": "Hi!"}
    assert to_messages(record, "alpaca") == [
        {"role": "user", "content": "Say hi"},
        {"role": "assistant", "content": "Hi!"},
    ]


def test_alpaca_with_input_appends_to_prompt() -> None:
    record = {"instruction": "Add", "input": "2 + 2", "output": "4"}
    messages = to_messages(record, "alpaca")
    assert messages[0] == {"role": "user", "content": "Add\n\n2 + 2"}


def test_alpaca_missing_output_raises() -> None:
    with pytest.raises(DatasetError, match="missing keys"):
        to_messages({"instruction": "Say hi"}, "alpaca")


def test_sharegpt_role_mapping() -> None:
    record = {
        "conversations": [
            {"from": "system", "value": "Be brief."},
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello"},
        ]
    }
    assert [m["role"] for m in to_messages(record, "sharegpt")] == [
        "system",
        "user",
        "assistant",
    ]


def test_sharegpt_unknown_speaker_raises() -> None:
    record = {"conversations": [{"from": "narrator", "value": "..."}]}
    with pytest.raises(DatasetError, match="unknown speaker"):
        to_messages(record, "sharegpt")


def test_sharegpt_empty_conversations_raises() -> None:
    with pytest.raises(DatasetError, match="no 'conversations' list"):
        to_messages({"conversations": []}, "sharegpt")


def test_openai_passthrough() -> None:
    record = {"messages": [{"role": "user", "content": "Hi"}]}
    assert to_messages(record, "openai") == [{"role": "user", "content": "Hi"}]


def test_openai_message_missing_role_raises() -> None:
    record = {"messages": [{"content": "Hi"}]}
    with pytest.raises(DatasetError, match="'role' and 'content'"):
        to_messages(record, "openai")


def test_completion_has_no_message_form() -> None:
    with pytest.raises(DatasetError, match="no message representation"):
        to_messages({"text": "raw"}, "completion")


# --- inspect_local_dataset ------------------------------------------------


def make_config(path: Path | str, **overrides: Any) -> DatasetConfig:
    settings: dict[str, Any] = {"path": str(path), "format": "alpaca"}
    if overrides.get("format") == "completion":
        settings["chat_template"] = None
    settings.update(overrides)
    return DatasetConfig.model_validate(settings)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def test_hub_id_is_skipped() -> None:
    assert inspect_local_dataset(make_config("yahma/alpaca-cleaned")) == []


def test_missing_file_reported(tmp_path: Path) -> None:
    problems = inspect_local_dataset(make_config(tmp_path / "nope.jsonl"))
    assert problems and "does not exist" in problems[0]


def test_valid_jsonl_passes(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    write_jsonl(dataset, [{"instruction": "a", "output": "b"}])
    assert inspect_local_dataset(make_config(dataset)) == []


def test_valid_json_array_passes(tmp_path: Path) -> None:
    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps([{"instruction": "a", "output": "b"}]), encoding="utf-8")
    assert inspect_local_dataset(make_config(dataset)) == []


def test_json_top_level_object_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps({"instruction": "a"}), encoding="utf-8")
    problems = inspect_local_dataset(make_config(dataset))
    assert problems and "must be a list" in problems[0]


def test_malformed_jsonl_reported_not_raised(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{not json}\n", encoding="utf-8")
    problems = inspect_local_dataset(make_config(dataset))
    assert problems and "could not read" in problems[0]


def test_empty_file_reported(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("", encoding="utf-8")
    problems = inspect_local_dataset(make_config(dataset))
    assert problems and "no records" in problems[0]


def test_format_mismatch_reported_per_record(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    write_jsonl(
        dataset,
        [
            {"instruction": "ok", "output": "ok"},
            {"messages": [{"role": "user", "content": "wrong format"}]},
        ],
    )
    problems = inspect_local_dataset(make_config(dataset))
    assert len(problems) == 1
    assert problems[0].startswith("record 1")


def test_csv_records_checked(tmp_path: Path) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text("instruction,output\na,b\n", encoding="utf-8")
    assert inspect_local_dataset(make_config(dataset)) == []


def test_completion_checks_text_field(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    write_jsonl(dataset, [{"body": "raw text"}])
    config = make_config(dataset, format="completion", text_field="text")
    problems = inspect_local_dataset(config)
    assert problems and "text_field" in problems[0]
    assert inspect_local_dataset(make_config(dataset, format="completion", text_field="body")) == []


def test_parquet_noted_not_failed(tmp_path: Path) -> None:
    dataset = tmp_path / "data.parquet"
    dataset.write_bytes(b"PAR1")
    problems = inspect_local_dataset(make_config(dataset))
    assert len(problems) == 1 and problems[0].startswith("note:")


def test_non_object_record_reported(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('["a", "list"]\n', encoding="utf-8")
    problems = inspect_local_dataset(make_config(dataset))
    assert problems and "not an object" in problems[0]
