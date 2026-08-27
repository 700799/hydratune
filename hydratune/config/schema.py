"""Pydantic v2 configuration schema for HydraTune.

Every ``config.yaml`` consumed by the CLI is parsed into
:class:`HydraTuneConfig`. All sections forbid unknown keys, so a typo like
``lora_droput`` fails validation loudly instead of being silently ignored.

The schema is the single source of truth for what a valid run looks like;
the trainer only translates an already-validated config into TRL/PEFT
objects and never re-checks invariants enforced here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)

AdapterType = Literal["lora", "qlora"]
DatasetFormat = Literal["alpaca", "sharegpt", "openai", "completion"]
ChatTemplate = Literal["tokenizer_default", "chatml", "llama3", "mistral", "zephyr"]
LrSchedulerType = Literal[
    "linear",
    "cosine",
    "cosine_with_restarts",
    "constant",
    "constant_with_warmup",
]
OptimizerType = Literal[
    "adamw_torch",
    "adamw_torch_fused",
    "adamw_8bit",
    "paged_adamw_8bit",
    "paged_adamw_32bit",
]
ReportTarget = Literal["wandb", "tensorboard", "none"]
QuantDtype = Literal["nf4", "fp4"]

#: Dataset formats whose records are (or can be converted to) role/content
#: message lists, and therefore go through a chat template at render time.
CONVERSATIONAL_FORMATS: frozenset[str] = frozenset({"alpaca", "sharegpt", "openai"})


class StrictModel(BaseModel):
    """Base for all config sections: unknown keys are errors, values are frozen-ish."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class BaseModelConfig(StrictModel):
    """Which model to fine-tune and how to load it."""

    name: str = Field(
        description="Hugging Face Hub id (e.g. 'meta-llama/Meta-Llama-3-8B') or a local path.",
        min_length=1,
    )
    revision: str | None = Field(
        default=None,
        description="Optional git revision (branch, tag, or commit) of the Hub repo.",
    )
    tokenizer_name: str | None = Field(
        default=None,
        description="Override tokenizer to load; defaults to the model's own tokenizer.",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Allow executing custom modeling code shipped with the Hub repo.",
    )


class DatasetConfig(StrictModel):
    """Where the training data lives and how its records are shaped."""

    path: str = Field(
        description="Hugging Face Hub dataset id or a local file (.json/.jsonl/.csv/.parquet).",
        min_length=1,
    )
    format: DatasetFormat = Field(
        default="alpaca",
        description=(
            "Record layout: 'alpaca' (instruction/input/output), 'sharegpt' "
            "(conversations with from/value turns), 'openai' (messages with "
            "role/content), or 'completion' (raw text)."
        ),
    )
    chat_template: ChatTemplate | None = Field(
        default="tokenizer_default",
        description=(
            "Chat template used to render conversational records into token "
            "streams. Ignored (and must be null) for 'completion' format."
        ),
    )
    split: str = Field(default="train", description="Dataset split to train on.")
    text_field: str = Field(
        default="text",
        description="Column containing raw text; only used with 'completion' format.",
    )
    validation_split: float | None = Field(
        default=None,
        gt=0.0,
        lt=0.5,
        description="Fraction of the train split held out for evaluation (e.g. 0.05).",
    )
    max_samples: int | None = Field(
        default=None,
        ge=1,
        description="Optional cap on the number of training records (useful for smoke tests).",
    )
    shuffle: bool = Field(default=True, description="Shuffle records before training.")

    @model_validator(mode="after")
    def _check_template_matches_format(self) -> DatasetConfig:
        if self.format == "completion" and self.chat_template is not None:
            raise ValueError(
                "dataset.chat_template does not apply to format='completion'; set it to null"
            )
        if self.format in CONVERSATIONAL_FORMATS and self.chat_template is None:
            raise ValueError(
                f"dataset.chat_template is required for conversational format "
                f"'{self.format}' (use 'tokenizer_default' to keep the model's own template)"
            )
        return self

    @property
    def is_local_file(self) -> bool:
        """True when ``path`` points at a file on disk rather than a Hub dataset id."""
        return Path(self.path).suffix.lower() in {".json", ".jsonl", ".csv", ".parquet"}


class TrainingConfig(StrictModel):
    """Optimization hyperparameters and run bookkeeping."""

    output_dir: Path = Field(
        default=Path("./outputs"),
        description="Directory for checkpoints, logs, and the final adapter/model.",
    )
    learning_rate: float = Field(default=2e-4, gt=0.0, le=1.0)
    epochs: float = Field(
        default=1.0,
        gt=0.0,
        description="Number of passes over the dataset; ignored when max_steps is set.",
    )
    max_steps: int | None = Field(
        default=None,
        ge=1,
        description="Hard cap on optimizer steps; overrides epochs when set.",
    )
    batch_size: int = Field(default=1, ge=1, description="Micro batch size per device.")
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    warmup_steps: int = Field(default=0, ge=0)
    warmup_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Warmup as a fraction of total steps; mutually exclusive with warmup_steps.",
    )
    lr_scheduler: LrSchedulerType = Field(default="cosine")
    optimizer: OptimizerType = Field(default="adamw_torch")
    weight_decay: float = Field(default=0.0, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    max_seq_length: int = Field(
        default=2048,
        ge=8,
        description="Maximum sequence length in tokens after packing/truncation.",
    )
    packing: bool = Field(
        default=True,
        description="Pack multiple short examples into each max_seq_length window.",
    )
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=500, ge=1)
    eval_steps: int | None = Field(
        default=None,
        ge=1,
        description="Evaluation cadence; requires dataset.validation_split.",
    )
    save_total_limit: int | None = Field(default=3, ge=1)
    seed: int = Field(default=42)
    resume_from_checkpoint: Path | None = Field(
        default=None,
        description="Checkpoint directory to resume from.",
    )
    report_to: ReportTarget = Field(default="none")

    @model_validator(mode="after")
    def _check_warmup_exclusive(self) -> TrainingConfig:
        if self.warmup_steps > 0 and self.warmup_ratio > 0.0:
            raise ValueError(
                "training.warmup_steps and training.warmup_ratio are mutually "
                "exclusive; set at most one of them"
            )
        return self

    @property
    def effective_batch_size(self) -> int:
        """Per-device effective batch size (micro batch x accumulation)."""
        return self.batch_size * self.gradient_accumulation_steps


class QuantizationConfig(StrictModel):
    """bitsandbytes 4-bit quantization settings; only meaningful for QLoRA."""

    bnb_4bit_quant_type: QuantDtype = Field(default="nf4")
    bnb_4bit_use_double_quant: bool = Field(
        default=True,
        description="Quantize the quantization constants for extra memory savings.",
    )


class PeftConfig(StrictModel):
    """Parameter-efficient fine-tuning (LoRA / QLoRA) settings.

    Omit the entire ``peft`` section to run full-parameter fine-tuning.
    """

    adapter: AdapterType = Field(
        description="'lora' trains adapters on the full-precision model; "
        "'qlora' additionally quantizes the base model to 4-bit."
    )
    r: int = Field(default=16, ge=1, le=1024, description="LoRA rank.")
    lora_alpha: int = Field(default=32, ge=1, description="LoRA scaling numerator.")
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: list[str] | Literal["all-linear"] = Field(
        default="all-linear",
        description="Module names to wrap with LoRA, or 'all-linear' to auto-detect.",
    )
    bias: Literal["none", "all", "lora_only"] = Field(default="none")
    use_rslora: bool = Field(
        default=False,
        description="Use rank-stabilized LoRA scaling (alpha / sqrt(r)).",
    )
    quantization: QuantizationConfig | None = Field(
        default=None,
        description="4-bit quantization details; only valid with adapter='qlora'.",
    )

    @model_validator(mode="after")
    def _check_quantization_matches_adapter(self) -> PeftConfig:
        if self.adapter == "qlora" and self.quantization is None:
            self.quantization = QuantizationConfig()
        elif self.adapter == "lora" and self.quantization is not None:
            raise ValueError(
                "peft.quantization is only valid with adapter='qlora'; "
                "remove it or switch the adapter type"
            )
        return self

    @model_validator(mode="after")
    def _check_target_modules_not_empty(self) -> PeftConfig:
        if isinstance(self.target_modules, list) and not self.target_modules:
            raise ValueError(
                "peft.target_modules must not be an empty list; use 'all-linear' "
                "to auto-detect linear layers"
            )
        return self


class HardwareConfig(StrictModel):
    """Precision and memory/compute trade-offs."""

    bf16: bool = Field(
        default=True,
        description="Train in bfloat16 (requires Ampere or newer GPUs).",
    )
    fp16: bool = Field(default=False, description="Train in float16.")
    flash_attention: bool = Field(
        default=False,
        description="Use FlashAttention-2 (requires the flash-attn package).",
    )
    gradient_checkpointing: bool = Field(
        default=True,
        description="Trade compute for memory by recomputing activations backward.",
    )
    tf32: bool = Field(
        default=True,
        description="Allow TF32 matmuls on Ampere+ GPUs (no effect elsewhere).",
    )

    @model_validator(mode="after")
    def _check_precision_exclusive(self) -> HardwareConfig:
        if self.bf16 and self.fp16:
            raise ValueError("hardware.bf16 and hardware.fp16 are mutually exclusive")
        return self


class HydraTuneConfig(StrictModel):
    """Root of the ``config.yaml`` schema."""

    base_model: BaseModelConfig
    dataset: DatasetConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    peft: PeftConfig | None = Field(
        default=None,
        description="LoRA/QLoRA settings; omit for full fine-tuning.",
    )
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)

    @model_validator(mode="after")
    def _check_cross_section_invariants(self) -> HydraTuneConfig:
        if self.training.eval_steps is not None and self.dataset.validation_split is None:
            raise ValueError(
                "training.eval_steps requires dataset.validation_split to be set"
            )
        if (
            self.peft is not None
            and self.peft.adapter == "qlora"
            and not (self.hardware.bf16 or self.hardware.fp16)
        ):
            raise ValueError(
                "QLoRA requires a half-precision compute dtype: enable "
                "hardware.bf16 (recommended) or hardware.fp16"
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def torch_dtype(self) -> Literal["bfloat16", "float16", "float32"]:
        """The torch dtype implied by the hardware precision flags."""
        if self.hardware.bf16:
            return "bfloat16"
        if self.hardware.fp16:
            return "float16"
        return "float32"

    @property
    def is_quantized(self) -> bool:
        """True when the base model will be loaded in 4-bit (QLoRA)."""
        return self.peft is not None and self.peft.adapter == "qlora"
