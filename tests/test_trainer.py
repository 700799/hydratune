"""Tests for the pure-logic pieces of the trainer (no torch/TRL required)."""

from __future__ import annotations

from hydratune.config.schema import TrainingConfig
from hydratune.training.trainer import resolve_warmup_steps


def test_explicit_warmup_steps_pass_through() -> None:
    training = TrainingConfig(warmup_steps=100)
    assert resolve_warmup_steps(training, num_train_records=10_000) == 100


def test_zero_warmup_by_default() -> None:
    assert resolve_warmup_steps(TrainingConfig(), num_train_records=10_000) == 0


def test_warmup_ratio_converts_to_steps() -> None:
    # 10_000 records / (4 batch * 2 accumulation) = 1250 updates/epoch, 2 epochs
    # = 2500 updates; 3% warmup = 75 steps.
    training = TrainingConfig(
        warmup_ratio=0.03, batch_size=4, gradient_accumulation_steps=2, epochs=2
    )
    assert resolve_warmup_steps(training, num_train_records=10_000) == 75


def test_warmup_ratio_respects_max_steps() -> None:
    training = TrainingConfig(warmup_ratio=0.1, max_steps=500)
    assert resolve_warmup_steps(training, num_train_records=1_000_000) == 50


def test_warmup_ratio_yields_at_least_one_step() -> None:
    training = TrainingConfig(warmup_ratio=0.01, batch_size=2)
    assert resolve_warmup_steps(training, num_train_records=5) == 1
