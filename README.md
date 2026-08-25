# HydraTune

YAML-configurable LLM fine-tuning harness, inspired by [Axolotl](https://github.com/axolotl-ai-cloud/axolotl).
Train, fine-tune, and evaluate open-weights models with a single human-readable
`config.yaml` — no boilerplate training code, full control over datasets,
LoRA/QLoRA parameters, and hardware settings.

Built on Hugging Face [TRL](https://github.com/huggingface/trl) (SFTTrainer),
[PEFT](https://github.com/huggingface/peft), and
[Accelerate](https://github.com/huggingface/accelerate), with a strict
[Pydantic v2](https://docs.pydantic.dev/) schema so config typos fail loudly
before any GPU time is spent.

## Quickstart

```bash
# Base install: config validation + CLI only (no CUDA required)
pip install -e .

# Full install with training dependencies
pip install -e ".[train]"

# Sanity-check a config and its dataset without touching a GPU
hydratune validate --config example_config.yaml

# Run the fine-tuning job
hydratune train --config example_config.yaml

# Multi-GPU via Accelerate
accelerate launch -m hydratune.cli train --config example_config.yaml
```

See [`example_config.yaml`](example_config.yaml) for a complete QLoRA recipe
for Llama-3-8B that fits on a single 24 GB GPU.

## CLI

| Command              | Status  | Purpose                                                     |
| -------------------- | ------- | ----------------------------------------------------------- |
| `hydratune train`    | working | Fine-tune per the config (SFT with optional LoRA/QLoRA).    |
| `hydratune validate` | working | Validate the config schema and sample local dataset files.  |
| `hydratune export`   | planned | Merge adapters into the base model, convert to GGUF.        |

All commands take `--config / -c <path>`.

## Configuration

A config file has five sections; only `base_model` and `dataset` are required.

```yaml
base_model:        # what to fine-tune
  name: meta-llama/Meta-Llama-3-8B

dataset:           # where the data lives and how records are shaped
  path: yahma/alpaca-cleaned          # Hub id or local .jsonl/.json/.csv/.parquet
  format: alpaca                      # alpaca | sharegpt | openai | completion
  chat_template: llama3               # tokenizer_default | chatml | llama3 | mistral | zephyr

training:          # optimization hyperparameters
  learning_rate: 2.0e-4
  epochs: 3
  batch_size: 2
  warmup_steps: 100

peft:              # omit this section entirely for full fine-tuning
  adapter: qlora                      # lora | qlora
  r: 32
  lora_alpha: 16
  target_modules: [q_proj, k_proj, v_proj, o_proj]

hardware:          # precision and memory trade-offs
  bf16: true
  flash_attention: true
  gradient_checkpointing: true
```

The full schema — every field, type, bound, and cross-field rule — lives in
[`hydratune/config/schema.py`](hydratune/config/schema.py). Unknown keys are
rejected, so a misspelled field is a validation error, not a silent no-op.

## Project layout

```
hydratune/
├── hydratune/
│   ├── cli.py                  # Typer entry point: train / validate / export
│   ├── config/
│   │   ├── schema.py           # Pydantic v2 schema (source of truth for config.yaml)
│   │   └── loader.py           # YAML -> HydraTuneConfig with readable errors
│   ├── data/
│   │   ├── formatting.py       # dataset formats, record -> messages, cheap sanity checks
│   │   └── chat_templates.py   # named Jinja chat templates
│   ├── training/
│   │   └── trainer.py          # config -> SFTTrainer (lazy heavy imports)
│   ├── export/                 # planned: adapter merge + GGUF export
│   └── utils/
│       └── errors.py           # HydraTuneError hierarchy (incl. OOM hints)
├── tests/
├── example_config.yaml         # Llama-3-8B QLoRA recipe
└── pyproject.toml
```

## Design notes

- **Light by default.** The base install depends only on `typer`, `pydantic`,
  `PyYAML`, and `rich`; torch/TRL/PEFT are behind the `[train]` extra and
  imported lazily, so you can validate configs on a laptop before shipping
  them to a GPU box.
- **Schema is the contract.** All invariants (bf16/fp16 exclusivity, QLoRA
  quantization pairing, warmup exclusivity, eval requiring a validation
  split) are enforced in `schema.py`; the trainer just maps fields onto
  library objects.
- **Graceful failures.** Config errors print as a lint-style report, OOMs are
  caught and re-raised with concrete remediation hints, and chat-template
  mismatches are detected before training starts.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## License

Apache-2.0
