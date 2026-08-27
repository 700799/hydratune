"""Dataset format definitions and lightweight sanity checks.

Two audiences use this module:

* ``hydratune validate`` calls :func:`inspect_local_dataset`, which peeks at
  the first few records of a local file using only the standard library, so
  configs can be sanity-checked on machines without the training extras.
* The trainer calls :func:`to_messages` per record to normalize every
  conversational format into OpenAI-style ``[{role, content}, ...]`` lists
  before the chat template renders them.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from hydratune.config.schema import DatasetConfig
from hydratune.utils.errors import DatasetError

#: Top-level keys each format requires on every record.
REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "alpaca": ("instruction", "output"),
    "sharegpt": ("conversations",),
    "openai": ("messages",),
    "completion": (),  # checked dynamically against dataset.text_field
}

#: Maps ShareGPT speaker tags to OpenAI-style roles.
_SHAREGPT_ROLES: dict[str, str] = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
}


def to_messages(record: dict[str, Any], format: str) -> list[dict[str, str]]:
    """Convert one raw record into an OpenAI-style message list.

    Raises:
        DatasetError: if the record does not match the declared format.
    """
    if format == "alpaca":
        missing = [key for key in REQUIRED_KEYS["alpaca"] if key not in record]
        if missing:
            raise DatasetError(f"alpaca record is missing keys {missing}: {record!r:.200}")
        prompt = str(record["instruction"])
        extra_input = str(record.get("input") or "").strip()
        if extra_input:
            prompt = f"{prompt}\n\n{extra_input}"
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": str(record["output"])},
        ]

    if format == "sharegpt":
        turns = record.get("conversations")
        if not isinstance(turns, list) or not turns:
            raise DatasetError(f"sharegpt record has no 'conversations' list: {record!r:.200}")
        messages: list[dict[str, str]] = []
        for turn in turns:
            speaker = str(turn.get("from", ""))
            role = _SHAREGPT_ROLES.get(speaker)
            if role is None:
                raise DatasetError(
                    f"sharegpt turn has unknown speaker {speaker!r} "
                    f"(expected one of {sorted(_SHAREGPT_ROLES)})"
                )
            messages.append({"role": role, "content": str(turn.get("value", ""))})
        return messages

    if format == "openai":
        messages_raw = record.get("messages")
        if not isinstance(messages_raw, list) or not messages_raw:
            raise DatasetError(f"openai record has no 'messages' list: {record!r:.200}")
        for message in messages_raw:
            if "role" not in message or "content" not in message:
                raise DatasetError(
                    f"openai message needs 'role' and 'content' keys: {message!r:.200}"
                )
        return [
            {"role": str(m["role"]), "content": str(m["content"])} for m in messages_raw
        ]

    raise DatasetError(f"Format {format!r} has no message representation")


def _iter_head_records(path: Path, limit: int) -> Iterator[dict[str, Any]]:
    """Yield up to ``limit`` records from a local .json/.jsonl/.csv file."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if i >= limit:
                    return
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise DatasetError(f"{path}: top-level JSON must be a list of records")
        yield from data[:limit]
    elif suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            for i, row in enumerate(csv.DictReader(handle)):
                if i >= limit:
                    return
                yield dict(row)
    else:
        raise DatasetError(f"Cannot inspect {suffix!r} files without training extras")


def inspect_local_dataset(config: DatasetConfig, sample_size: int = 5) -> list[str]:
    """Cheaply sanity-check a local dataset file against its declared format.

    Returns a list of human-readable problems; an empty list means the sample
    looked consistent. Hub datasets and formats we cannot read with the
    standard library are skipped (with a note) rather than failed.
    """
    if not config.is_local_file:
        return []

    path = Path(config.path)
    if not path.is_file():
        return [f"dataset.path does not exist: {path}"]
    if path.suffix.lower() == ".parquet":
        return [f"note: skipping content check for {path.name} (parquet needs training extras)"]

    problems: list[str] = []
    try:
        records = list(_iter_head_records(path, sample_size))
    except (json.JSONDecodeError, csv.Error, DatasetError, UnicodeDecodeError, OSError) as exc:
        return [f"could not read {path.name}: {exc}"]

    if not records:
        return [f"{path.name} contains no records"]

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            problems.append(f"record {index} is not an object: {record!r:.120}")
            continue
        if config.format == "completion":
            if config.text_field not in record:
                problems.append(
                    f"record {index} is missing text_field {config.text_field!r} "
                    f"(has keys: {sorted(record)})"
                )
            continue
        try:
            to_messages(record, config.format)
        except DatasetError as exc:
            problems.append(f"record {index}: {exc}")

    return problems
