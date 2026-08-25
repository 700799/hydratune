"""Translate a validated :class:`HydraTuneConfig` into a TRL ``SFTTrainer`` run.

This module is only imported by the CLI when ``hydratune train`` actually
runs, so the base install (config + validate) never needs torch, TRL, or
CUDA. All heavy third-party imports therefore live inside functions.

The config schema has already enforced every invariant (precision flags,
adapter/quantization pairing, warmup exclusivity); this module only maps
fields onto library objects and adds the runtime checks that need a live
tokenizer, dataset, or GPU: chat-template compatibility and OOM handling.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hydratune.config.schema import HydraTuneConfig, TrainingConfig
from hydratune.data.chat_templates import resolve_chat_template
from hydratune.data.formatting import to_messages
from hydratune.utils.errors import ChatTemplateError, DatasetError, OOMRiskError

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids hard deps
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

#: Remedies surfaced with every OOM error, roughly in order of cheapness.
_OOM_HINTS: list[str] = [
    "Lower training.batch_size (raise gradient_accumulation_steps to keep the effective size)",
    "Enable hardware.gradient_checkpointing",
    "Reduce training.max_seq_length",
    "Switch peft.adapter to 'qlora' to load the base model in 4-bit",
    "Use a paged optimizer (training.optimizer: paged_adamw_8bit)",
]


def load_tokenizer(config: HydraTuneConfig) -> PreTrainedTokenizerBase:
    """Load the tokenizer and apply/verify the configured chat template."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model.tokenizer_name or config.base_model.name,
        revision=config.base_model.revision,
        trust_remote_code=config.base_model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    template = config.dataset.chat_template
    if template == "tokenizer_default":
        if tokenizer.chat_template is None:
            raise ChatTemplateError(
                f"Tokenizer for {config.base_model.name!r} ships no chat template, "
                "but dataset.chat_template is 'tokenizer_default'. Pick an explicit "
                "template (e.g. 'chatml' or 'llama3') in the config."
            )
    elif template is not None:
        tokenizer.chat_template = resolve_chat_template(template)
    return tokenizer


def load_datasets(
    config: HydraTuneConfig, tokenizer: PreTrainedTokenizerBase
) -> tuple[Dataset, Dataset | None]:
    """Load, format, and optionally split the dataset into (train, eval)."""
    from datasets import load_dataset

    dataset_config = config.dataset
    if dataset_config.is_local_file:
        suffix = Path(dataset_config.path).suffix.lower().lstrip(".")
        loader_name = "json" if suffix == "jsonl" else suffix
        dataset = load_dataset(loader_name, data_files=dataset_config.path, split="train")
    else:
        dataset = load_dataset(dataset_config.path, split=dataset_config.split)

    if dataset_config.shuffle:
        dataset = dataset.shuffle(seed=config.training.seed)
    if dataset_config.max_samples is not None:
        dataset = dataset.select(range(min(dataset_config.max_samples, len(dataset))))

    if dataset_config.format == "completion":
        if dataset_config.text_field not in dataset.column_names:
            raise DatasetError(
                f"dataset.text_field {dataset_config.text_field!r} not found in "
                f"columns {dataset.column_names}"
            )
        if dataset_config.text_field != "text":
            dataset = dataset.rename_column(dataset_config.text_field, "text")
    else:

        def render(record: dict[str, Any]) -> dict[str, str]:
            messages = to_messages(record, dataset_config.format)
            return {
                "text": tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            }

        dataset = dataset.map(render, remove_columns=dataset.column_names, desc="Rendering")

    if dataset_config.validation_split is not None:
        split = dataset.train_test_split(
            test_size=dataset_config.validation_split, seed=config.training.seed
        )
        return split["train"], split["test"]
    return dataset, None


def load_model(config: HydraTuneConfig) -> PreTrainedModel:
    """Load the base model with the configured precision/quantization."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    torch_dtype = getattr(torch, config.torch_dtype)
    model_kwargs: dict[str, Any] = {
        "revision": config.base_model.revision,
        "trust_remote_code": config.base_model.trust_remote_code,
        "dtype": torch_dtype,
    }
    if config.hardware.flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    if config.is_quantized:
        assert config.peft is not None and config.peft.quantization is not None
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.peft.quantization.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.peft.quantization.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=torch_dtype,
        )

    if config.hardware.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        return AutoModelForCausalLM.from_pretrained(config.base_model.name, **model_kwargs)
    except torch.cuda.OutOfMemoryError as exc:
        raise OOMRiskError(
            f"Ran out of GPU memory while loading {config.base_model.name!r}.",
            hints=_OOM_HINTS,
        ) from exc


def build_peft_config(config: HydraTuneConfig) -> Any | None:
    """Build a peft.LoraConfig, or None for full fine-tuning."""
    from peft import LoraConfig

    if config.peft is None:
        return None
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=config.peft.r,
        lora_alpha=config.peft.lora_alpha,
        lora_dropout=config.peft.lora_dropout,
        target_modules=config.peft.target_modules,
        bias=config.peft.bias,
        use_rslora=config.peft.use_rslora,
    )


def resolve_warmup_steps(training: TrainingConfig, num_train_records: int) -> int:
    """Translate ``warmup_ratio`` into a concrete step count.

    Transformers 5 dropped ``warmup_ratio`` from ``TrainingArguments``, so the
    ratio is converted here from the estimated total optimizer updates. With
    ``packing`` enabled the record count overestimates the number of packed
    sequences, so the resulting warmup errs slightly long — acceptable for a
    warmup heuristic.
    """
    if training.warmup_ratio == 0.0:
        return training.warmup_steps
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    records_per_update = training.batch_size * training.gradient_accumulation_steps * world_size
    updates_per_epoch = math.ceil(num_train_records / records_per_update)
    total_updates = (
        training.max_steps
        if training.max_steps is not None
        else math.ceil(updates_per_epoch * training.epochs)
    )
    return max(1, round(total_updates * training.warmup_ratio))


def run_training(config: HydraTuneConfig) -> Path:
    """Execute a full SFT run and return the output directory."""
    import torch
    from trl import SFTConfig, SFTTrainer

    tokenizer = load_tokenizer(config)
    train_dataset, eval_dataset = load_datasets(config, tokenizer)
    model = load_model(config)

    training = config.training
    sft_config = SFTConfig(
        output_dir=str(training.output_dir),
        learning_rate=training.learning_rate,
        num_train_epochs=training.epochs,
        max_steps=training.max_steps if training.max_steps is not None else -1,
        per_device_train_batch_size=training.batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        warmup_steps=resolve_warmup_steps(training, len(train_dataset)),
        lr_scheduler_type=training.lr_scheduler,
        optim=training.optimizer,
        weight_decay=training.weight_decay,
        max_grad_norm=training.max_grad_norm,
        max_length=training.max_seq_length,
        packing=training.packing,
        dataset_text_field="text",
        logging_steps=training.logging_steps,
        save_steps=training.save_steps,
        save_total_limit=training.save_total_limit,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=training.eval_steps,
        seed=training.seed,
        bf16=config.hardware.bf16,
        fp16=config.hardware.fp16,
        gradient_checkpointing=config.hardware.gradient_checkpointing,
        report_to=training.report_to,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=build_peft_config(config),
    )

    try:
        trainer.train(
            resume_from_checkpoint=(
                str(training.resume_from_checkpoint)
                if training.resume_from_checkpoint is not None
                else None
            )
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise OOMRiskError("Ran out of GPU memory during training.", hints=_OOM_HINTS) from exc

    trainer.save_model(str(training.output_dir))
    tokenizer.save_pretrained(str(training.output_dir))
    return training.output_dir
