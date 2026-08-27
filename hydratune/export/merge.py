"""Merge trained LoRA/QLoRA adapters back into their base model.

The result is a standalone model directory that loads with a plain
``AutoModelForCausalLM.from_pretrained`` — no PEFT at inference time.

As with :mod:`hydratune.training.trainer`, every heavy import lives inside a
function so the base install (config + CLI) stays free of torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from hydratune.config.schema import HydraTuneConfig
from hydratune.utils.errors import ExportError

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids hard deps
    from transformers import PreTrainedTokenizerBase

#: Written by PEFT next to the adapter weights; its presence is how we tell a
#: real adapter directory from an empty or mistyped path.
ADAPTER_CONFIG_FILE = "adapter_config.json"

#: Files a tokenizer save produces; any one of them means the adapter directory
#: carries its own tokenizer and we should prefer it over the base model's.
_TOKENIZER_MARKERS = ("tokenizer_config.json", "tokenizer.json")


def _resolve_dirs(
    config: HydraTuneConfig,
    adapter_dir: Path | None,
    output_dir: Path | None,
) -> tuple[Path, Path]:
    """Apply the defaults: adapter from the training output, merged beside it."""
    resolved_adapter = adapter_dir or config.training.output_dir
    resolved_output = output_dir or resolved_adapter / "merged"
    return Path(resolved_adapter), Path(resolved_output)


def _load_tokenizer_for_merge(
    config: HydraTuneConfig, adapter_dir: Path
) -> PreTrainedTokenizerBase:
    """Prefer the tokenizer saved with the adapter; fall back to the base model.

    Training saves the tokenizer alongside the adapter, and it may carry an
    overridden chat template — keeping it means the merged model renders
    prompts exactly the way it was trained.
    """
    from transformers import AutoTokenizer

    if any((adapter_dir / marker).is_file() for marker in _TOKENIZER_MARKERS):
        return AutoTokenizer.from_pretrained(str(adapter_dir))
    return AutoTokenizer.from_pretrained(
        config.base_model.tokenizer_name or config.base_model.name,
        revision=config.base_model.revision,
        trust_remote_code=config.base_model.trust_remote_code,
    )


def merge_adapter(
    config: HydraTuneConfig,
    adapter_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Merge the adapter in ``adapter_dir`` into the base model and save it.

    Args:
        config: A validated config; ``base_model`` identifies what to merge into.
        adapter_dir: Adapter location; defaults to ``training.output_dir``.
        output_dir: Where to write the merged model; defaults to
            ``<adapter_dir>/merged``.

    Returns:
        The directory the merged model was written to.

    Raises:
        ExportError: if the config declares no adapter, the adapter directory
            is missing or is not a PEFT adapter, or the output directory would
            overwrite the adapter itself.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if config.peft is None:
        raise ExportError(
            "This config has no 'peft' section, so there is no adapter to merge "
            "(a full fine-tune already produces a standalone model)."
        )

    resolved_adapter, resolved_output = _resolve_dirs(config, adapter_dir, output_dir)

    if not resolved_adapter.is_dir():
        raise ExportError(f"Adapter directory not found: {resolved_adapter}")
    if not (resolved_adapter / ADAPTER_CONFIG_FILE).is_file():
        raise ExportError(
            f"{resolved_adapter} does not look like a PEFT adapter "
            f"(no {ADAPTER_CONFIG_FILE}). Point --adapter at a directory produced "
            "by 'hydratune train'."
        )
    if resolved_output.resolve() == resolved_adapter.resolve():
        raise ExportError(
            "Refusing to write the merged model over the adapter directory; "
            "choose a different --output."
        )

    # Deliberately NOT trainer.load_model(): that applies 4-bit quantization for
    # QLoRA configs, and merging LoRA weights into NF4 weights loses precision.
    # The standard practice is to merge into a half/full precision base instead.
    model_kwargs: dict[str, Any] = {
        "revision": config.base_model.revision,
        "trust_remote_code": config.base_model.trust_remote_code,
        "dtype": getattr(torch, config.torch_dtype),
    }
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model.name, **model_kwargs)

    peft_model = PeftModel.from_pretrained(base_model, str(resolved_adapter))
    # merge_and_unload lives on the tuner (LoraModel) and is reached through
    # PeftModel.__getattr__ delegation.
    merged = peft_model.merge_and_unload()

    resolved_output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(resolved_output))
    _load_tokenizer_for_merge(config, resolved_adapter).save_pretrained(str(resolved_output))
    return resolved_output
