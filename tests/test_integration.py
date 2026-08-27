"""End-to-end integration tests: real SFTTrainer runs against a tiny local model.

These tests build a ~100k-parameter Llama-architecture checkpoint and a small
BPE tokenizer on the fly (no network), then drive the actual
:func:`hydratune.training.trainer.run_training` pipeline. They skip
automatically on a base install without the ``[train]`` extra, and each run
completes in a few seconds on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")
pytest.importorskip("datasets")

from hydratune.config.schema import HydraTuneConfig  # noqa: E402
from hydratune.export.merge import merge_adapter  # noqa: E402
from hydratune.training.trainer import run_training  # noqa: E402
from hydratune.utils.errors import ChatTemplateError, ExportError  # noqa: E402

CORPUS = [
    "What is the capital of France? The capital of France is Paris.",
    "Add the two numbers. 2 + 2 = 4.",
    "Translate to German. Good morning! Guten Morgen!",
    "Name a primary color. Red is a primary color.",
    "Write a haiku about rivers. A river whispers, carving valleys.",
    "Spell the word cat. c a t.",
]

RECORDS = [
    {"instruction": "Say hi", "input": "", "output": "Hi there!"},
    {"instruction": "Add", "input": "2 + 2", "output": "4"},
    {"instruction": "Translate to German", "input": "Good morning", "output": "Guten Morgen"},
    {"instruction": "Name a color", "input": "", "output": "Red"},
    {"instruction": "Count to three", "input": "", "output": "1 2 3"},
    {"instruction": "Spell cat", "input": "", "output": "c a t"},
]


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build and save a tiny Llama checkpoint + BPE tokenizer (no chat template)."""
    out = tmp_path_factory.mktemp("tiny-llama")

    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"]
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.train_from_iterator(
        CORPUS, tokenizers.trainers.BpeTrainer(vocab_size=512, special_tokens=specials)
    )
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|endoftext|>",
        eos_token="<|im_end|>",
        pad_token="<|pad|>",
    )
    fast.save_pretrained(out)

    config = transformers.LlamaConfig(
        vocab_size=fast.vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        bos_token_id=fast.bos_token_id,
        eos_token_id=fast.eos_token_id,
        pad_token_id=fast.pad_token_id,
    )
    transformers.LlamaForCausalLM(config).save_pretrained(out)
    return out


@pytest.fixture()
def dataset_file(tmp_path: Path) -> Path:
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in RECORDS), encoding="utf-8")
    return path


def make_config(
    tiny_model_dir: Path, dataset_file: Path, output_dir: Path, **overrides: Any
) -> HydraTuneConfig:
    settings: dict[str, Any] = {
        "base_model": {"name": str(tiny_model_dir)},
        "dataset": {
            "path": str(dataset_file),
            "format": "alpaca",
            "chat_template": "chatml",
        },
        "training": {
            "output_dir": str(output_dir),
            "epochs": 1,
            "batch_size": 2,
            "max_seq_length": 128,
            "packing": False,
            "logging_steps": 1,
            "report_to": "none",
        },
        "peft": {
            "adapter": "lora",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "hardware": {"bf16": False, "fp16": False, "gradient_checkpointing": False},
    }
    for section, values in overrides.items():
        settings[section].update(values)
    return HydraTuneConfig.model_validate(settings)


def test_training_produces_loadable_lora_adapter(
    tiny_model_dir: Path, dataset_file: Path, tmp_path: Path
) -> None:
    config = make_config(tiny_model_dir, dataset_file, tmp_path / "adapter")
    output_dir = run_training(config)

    adapter_config_path = output_dir / "adapter_config.json"
    assert adapter_config_path.is_file()
    assert (output_dir / "adapter_model.safetensors").is_file()

    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    assert adapter_config["r"] == 8
    assert adapter_config["lora_alpha"] == 16
    assert set(adapter_config["target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj"}

    # The tokenizer must ship with the adapter so inference can reload both.
    assert (output_dir / "tokenizer_config.json").is_file()


def test_training_with_eval_split_and_warmup_ratio(
    tiny_model_dir: Path, dataset_file: Path, tmp_path: Path
) -> None:
    config = make_config(
        tiny_model_dir,
        dataset_file,
        tmp_path / "adapter",
        dataset={"validation_split": 0.34},
        training={"eval_steps": 1, "warmup_ratio": 0.5, "warmup_steps": 0},
    )
    output_dir = run_training(config)
    assert (output_dir / "adapter_model.safetensors").is_file()


def test_merge_produces_standalone_model_with_changed_weights(
    tiny_model_dir: Path, dataset_file: Path, tmp_path: Path
) -> None:
    """Train an adapter, merge it, and prove the merge actually applied."""
    adapter_dir = tmp_path / "adapter"
    config = make_config(tiny_model_dir, dataset_file, adapter_dir)
    run_training(config)

    merged_dir = merge_adapter(config)
    assert merged_dir == adapter_dir / "merged"

    # The merged model must load without PEFT in the picture.
    merged = transformers.AutoModelForCausalLM.from_pretrained(str(merged_dir))
    base = transformers.AutoModelForCausalLM.from_pretrained(str(tiny_model_dir))
    assert (merged_dir / "config.json").is_file()
    assert (merged_dir / "tokenizer_config.json").is_file()

    # A LoRA target module's weights must differ from the base; an unadapted
    # module (the LM head is not in target_modules) must be untouched. Together
    # these show the merge applied the adapter rather than copying the base.
    merged_state, base_state = merged.state_dict(), base.state_dict()
    q_proj_key = "model.layers.0.self_attn.q_proj.weight"
    assert not torch.equal(merged_state[q_proj_key], base_state[q_proj_key])
    head_key = "model.embed_tokens.weight"
    assert torch.equal(merged_state[head_key], base_state[head_key])


def test_merge_rejects_directory_without_adapter(
    tiny_model_dir: Path, dataset_file: Path, tmp_path: Path
) -> None:
    empty = tmp_path / "not-an-adapter"
    empty.mkdir()
    config = make_config(tiny_model_dir, dataset_file, tmp_path / "unused")
    with pytest.raises(ExportError, match="does not look like a PEFT adapter"):
        merge_adapter(config, adapter_dir=empty)


def test_merge_requires_a_peft_section(
    tiny_model_dir: Path, dataset_file: Path, tmp_path: Path
) -> None:
    config = make_config(tiny_model_dir, dataset_file, tmp_path / "out")
    full_finetune = config.model_dump(exclude={"torch_dtype", "peft"})
    with pytest.raises(ExportError, match="no 'peft' section"):
        merge_adapter(HydraTuneConfig.model_validate(full_finetune))


def test_tokenizer_default_without_template_fails_fast(
    tiny_model_dir: Path, dataset_file: Path, tmp_path: Path
) -> None:
    # The tiny tokenizer ships no chat template, so 'tokenizer_default' must be
    # rejected before any training starts.
    config = make_config(
        tiny_model_dir,
        dataset_file,
        tmp_path / "adapter",
        dataset={"chat_template": "tokenizer_default"},
    )
    with pytest.raises(ChatTemplateError, match="ships no chat template"):
        run_training(config)
