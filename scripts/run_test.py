#!/usr/bin/env python3
"""End-to-end smoke test for the HydraTune training pipeline.

Reads ``configs/test_smollm2.yaml`` through the Pydantic v2 schema, writes a
5-row dummy alpaca-format JSONL dataset, fine-tunes SmolLM2-135M-Instruct
with LoRA via TRL's SFTTrainer, and saves the adapter to
``./output_test_adapter``. Runs on CPU (or any GPU) in well under 30 seconds
once the base model is in the local Hugging Face cache.

Usage, from the repo root:

    pip install -e ".[train]"
    python scripts/run_test.py [--config configs/test_smollm2.yaml]

Exit code 0 means the pipeline trained and produced adapter weights.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # allow running without `pip install -e .`

from hydratune.config import HydraTuneConfig, load_config  # noqa: E402
from hydratune.utils.errors import HydraTuneError  # noqa: E402

#: Five deterministic alpaca-format records — enough for a few optimizer steps.
DUMMY_RECORDS: list[dict[str, str]] = [
    {
        "instruction": "What is the capital of France?",
        "input": "",
        "output": "The capital of France is Paris.",
    },
    {
        "instruction": "Add the two numbers.",
        "input": "2 + 2",
        "output": "2 + 2 = 4.",
    },
    {
        "instruction": "Translate to German.",
        "input": "Good morning!",
        "output": "Guten Morgen!",
    },
    {
        "instruction": "Name a primary color.",
        "input": "",
        "output": "Red is a primary color.",
    },
    {
        "instruction": "Write a one-line haiku about rivers.",
        "input": "",
        "output": "A river whispers / carving valleys out of stone / patient as the years.",
    },
]

#: Files PEFT writes for a LoRA adapter; their presence is our pass criterion.
EXPECTED_ADAPTER_FILES: tuple[str, ...] = ("adapter_config.json", "adapter_model.safetensors")


def write_dummy_dataset(config: HydraTuneConfig) -> Path:
    """Create the dummy JSONL dataset at the path the config points to."""
    dataset_path = Path(config.dataset.path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        "\n".join(json.dumps(record) for record in DUMMY_RECORDS) + "\n",
        encoding="utf-8",
    )
    return dataset_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "test_smollm2.yaml",
        help="HydraTune YAML config to run (default: configs/test_smollm2.yaml).",
    )
    args = parser.parse_args()

    config = load_config(args.config)  # Pydantic v2 validation happens here
    dataset_path = write_dummy_dataset(config)
    print(f"Config OK: {args.config}")
    print(f"Dummy dataset: {dataset_path} ({len(DUMMY_RECORDS)} records)")

    from hydratune.training.trainer import run_training  # heavy imports live here

    start = time.perf_counter()
    output_dir = run_training(config)
    elapsed = time.perf_counter() - start

    missing = [name for name in EXPECTED_ADAPTER_FILES if not (output_dir / name).is_file()]
    if missing:
        print(f"FAIL: adapter files missing from {output_dir}: {missing}", file=sys.stderr)
        return 1

    adapter_size_kib = (output_dir / "adapter_model.safetensors").stat().st_size / 1024
    print(
        f"PASS: LoRA adapter saved to {output_dir} "
        f"({adapter_size_kib:.0f} KiB) in {elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HydraTuneError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
