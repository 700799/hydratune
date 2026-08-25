"""Schema validation tests: the invariants the trainer relies on."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from hydratune.config.schema import HydraTuneConfig


def minimal_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "base_model": {"name": "meta-llama/Meta-Llama-3-8B"},
        "dataset": {"path": "yahma/alpaca-cleaned", "format": "alpaca"},
    }
    config.update(overrides)
    return config


def test_minimal_config_validates_with_defaults() -> None:
    config = HydraTuneConfig.model_validate(minimal_config())
    assert config.training.learning_rate == 2e-4
    assert config.peft is None
    assert config.torch_dtype == "bfloat16"
    assert not config.is_quantized


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="lora_droput"):
        HydraTuneConfig.model_validate(
            minimal_config(peft={"adapter": "lora", "lora_droput": 0.1})
        )


def test_bf16_and_fp16_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        HydraTuneConfig.model_validate(
            minimal_config(hardware={"bf16": True, "fp16": True})
        )


def test_warmup_steps_and_ratio_are_mutually_exclusive() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        HydraTuneConfig.model_validate(
            minimal_config(training={"warmup_steps": 100, "warmup_ratio": 0.05})
        )


def test_qlora_gets_default_quantization() -> None:
    config = HydraTuneConfig.model_validate(minimal_config(peft={"adapter": "qlora"}))
    assert config.peft is not None
    assert config.peft.quantization is not None
    assert config.peft.quantization.bnb_4bit_quant_type == "nf4"
    assert config.is_quantized


def test_lora_rejects_quantization_block() -> None:
    with pytest.raises(ValidationError, match="only valid with adapter='qlora'"):
        HydraTuneConfig.model_validate(
            minimal_config(
                peft={"adapter": "lora", "quantization": {"bnb_4bit_quant_type": "nf4"}}
            )
        )


def test_qlora_requires_half_precision() -> None:
    with pytest.raises(ValidationError, match="half-precision"):
        HydraTuneConfig.model_validate(
            minimal_config(
                peft={"adapter": "qlora"},
                hardware={"bf16": False, "fp16": False},
            )
        )


def test_completion_format_rejects_chat_template() -> None:
    with pytest.raises(ValidationError, match="does not apply"):
        HydraTuneConfig.model_validate(
            minimal_config(
                dataset={
                    "path": "data.jsonl",
                    "format": "completion",
                    "chat_template": "chatml",
                }
            )
        )


def test_completion_format_accepts_null_template() -> None:
    config = HydraTuneConfig.model_validate(
        minimal_config(
            dataset={"path": "data.jsonl", "format": "completion", "chat_template": None}
        )
    )
    assert config.dataset.is_local_file


def test_eval_steps_requires_validation_split() -> None:
    with pytest.raises(ValidationError, match="validation_split"):
        HydraTuneConfig.model_validate(minimal_config(training={"eval_steps": 100}))


def test_learning_rate_bounds() -> None:
    with pytest.raises(ValidationError):
        HydraTuneConfig.model_validate(minimal_config(training={"learning_rate": 0}))
    with pytest.raises(ValidationError):
        HydraTuneConfig.model_validate(minimal_config(training={"learning_rate": 2.0}))


def test_empty_target_modules_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be an empty list"):
        HydraTuneConfig.model_validate(
            minimal_config(peft={"adapter": "lora", "target_modules": []})
        )
