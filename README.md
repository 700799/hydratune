# HydraTune

[![CI](https://github.com/700799/hydratune/actions/workflows/ci.yml/badge.svg)](https://github.com/700799/hydratune/actions/workflows/ci.yml)

**HydraTune is a YAML-configurable LLM fine-tuning harness**, inspired by
[Axolotl](https://github.com/axolotl-ai-cloud/axolotl). You describe a
training run — base model, dataset, hyperparameters, LoRA/QLoRA settings,
hardware trade-offs — in a single human-readable `config.yaml`, and HydraTune
turns it into a supervised fine-tuning run on Hugging Face
[TRL](https://github.com/huggingface/trl)'s `SFTTrainer`, with
[PEFT](https://github.com/huggingface/peft) adapters and
[Accelerate](https://github.com/huggingface/accelerate) for multi-GPU
execution.

No boilerplate training script. No silent misconfiguration: the config is
validated by a strict [Pydantic v2](https://docs.pydantic.dev/) schema before
any GPU time is spent, and a misspelled field is a hard error, not a no-op.

```yaml
base_model:
  name: meta-llama/Meta-Llama-3-8B
dataset:
  path: yahma/alpaca-cleaned
  format: alpaca
  chat_template: llama3
training:
  learning_rate: 2.0e-4
  epochs: 3
peft:
  adapter: qlora
  r: 32
```

```bash
hydratune validate --config config.yaml   # catch mistakes before GPU time
hydratune train    --config config.yaml   # run it
```

---

## Table of contents

- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [CLI reference](#cli-reference)
- [Configuration reference](#configuration-reference)
  - [`base_model`](#base_model)
  - [`dataset`](#dataset)
  - [`training`](#training)
  - [`peft`](#peft)
  - [`hardware`](#hardware)
  - [Cross-section rules](#cross-section-rules)
- [Dataset formats](#dataset-formats)
- [Chat templates](#chat-templates)
- [Recipes](#recipes)
- [Errors and troubleshooting](#errors-and-troubleshooting)
- [Architecture](#architecture)
- [Development](#development)
- [Roadmap](#roadmap)
- [License](#license)

---

## Features

- **One YAML file per run.** Everything that defines a fine-tune lives in one
  reviewable, diffable, version-controllable file.
- **Strict validation, early.** Every section rejects unknown keys
  (`extra="forbid"`), every field is typed and bounded, and cross-field
  rules (e.g. QLoRA requires half precision) are enforced in the schema —
  errors surface as a lint-style report, never a Python traceback.
- **LoRA and QLoRA out of the box.** Rank, alpha, dropout, target modules,
  rank-stabilized scaling, and bitsandbytes 4-bit quantization (NF4/FP4,
  double quantization) are all first-class config. Omit the `peft` section
  for full-parameter fine-tuning.
- **Dataset flexibility.** Alpaca, ShareGPT, and OpenAI-messages record
  formats plus raw-text completion data; Hub datasets or local
  `.jsonl`/`.json`/`.csv`/`.parquet` files; automatic conversion to a
  common message form and rendering through a chat template.
- **Token packing** via TRL, on by default, so short examples don't waste
  sequence budget.
- **Light by default.** The base install needs only `typer`, `pydantic`,
  `PyYAML`, and `rich` — no CUDA, no torch. `hydratune validate` runs on a
  laptop; heavy dependencies live behind the `[train]` extra and are
  imported lazily.
- **Fail fast, fail helpfully.** Chat-template mismatches are detected
  before training starts; CUDA OOM is caught and re-raised with concrete,
  ordered remediation hints; local datasets can be sanity-checked against
  their declared format without loading any ML library.
- **Typed and tested.** `mypy --strict` clean, `py.typed` shipped, and a
  test suite spanning schema invariants, CLI behavior, and real end-to-end
  training runs against a tiny local model.

## Installation

Requires Python 3.10+.

```bash
# Base install: CLI + config validation only (no CUDA, no torch)
pip install -e .

# Everything needed to actually train
pip install -e ".[train]"

# Development tools (pytest, ruff, mypy, ...)
pip install -e ".[dev]"
```

The `[train]` extra pulls in `torch`, `transformers`, `trl`, `peft`,
`accelerate`, `bitsandbytes`, `datasets`, and `sentencepiece`. The split
exists so you can author and validate configs on machines that will never
run the job — validation needs nothing heavy.

> **Note on FlashAttention:** `hardware.flash_attention: true` additionally
> requires the [`flash-attn`](https://github.com/Dao-AILab/flash-attention)
> package, which needs a CUDA toolchain to build. It is deliberately not part
> of the `[train]` extra; install it separately where supported.

## Quickstart

1. **Write a config** (or start from [`example_config.yaml`](example_config.yaml),
   a complete Llama-3-8B QLoRA recipe sized for a single 24 GB GPU):

2. **Validate it.** Schema checking plus — for local dataset files — a
   sample of records checked against the declared format:

   ```text
   $ hydratune validate --config example_config.yaml
   Config OK — schema validation passed.
                      HydraTune run summary
   ┌───────────────────┬──────────────────────────────────────┐
   │ Base model        │ meta-llama/Meta-Llama-3-8B           │
   │ Dataset           │ yahma/alpaca-cleaned (alpaca)        │
   │ Chat template     │ llama3                               │
   │ Adapter           │ qlora (r=32, alpha=16)               │
   │ Precision         │ bfloat16                             │
   │ Batch             │ 2 x 8 accumulation = 16 effective    │
   │ Sequence length   │ 4096                                 │
   │ Output dir        │ outputs/llama3-8b-qlora              │
   └───────────────────┴──────────────────────────────────────┘
   ```

3. **Train:**

   ```bash
   hydratune train --config example_config.yaml
   ```

4. **Multi-GPU** — HydraTune is Accelerate-native; launch the same command
   through `accelerate`:

   ```bash
   accelerate config          # once, to describe your machine
   accelerate launch -m hydratune.cli train --config example_config.yaml
   ```

The trained adapter (or full model), tokenizer, and checkpoints land in
`training.output_dir`.

## CLI reference

All commands take `--config` / `-c <path>` pointing at a YAML config file.

| Command              | Status  | Purpose                                                        |
| -------------------- | ------- | -------------------------------------------------------------- |
| `hydratune train`    | working | Run the fine-tuning job described by the config.               |
| `hydratune validate` | working | Validate the config schema; optionally sample local datasets.  |
| `hydratune export`   | planned | Merge adapters into the base model; convert to GGUF.           |

### `hydratune train --config <file>`

Prints the run summary, then loads the tokenizer, dataset, and model and
hands off to TRL's `SFTTrainer`. Training dependencies are imported at this
point — if they're missing you get a one-line pointer to
`pip install "hydratune[train]"` instead of an ImportError stack.

Exit codes: `0` success · `1` config error, dataset/template error, or OOM.

### `hydratune validate --config <file> [--check-dataset | --no-check-dataset]`

Validates the config against the schema and prints the run summary. With
`--check-dataset` (the default), local dataset files are additionally opened
and the first few records checked against the declared `dataset.format` —
using only the Python standard library, so this works on a base install.
Hub dataset ids are noted and skipped (they're checked at training time);
parquet files are noted as requiring the training extras.

Exit codes: `0` valid · `1` schema or dataset problems (each listed).

### `hydratune export --config <file>`

Not implemented yet; exits with code `2` and a message. See
[Roadmap](#roadmap).

### Global flags

`hydratune --version` prints the installed version.

## Configuration reference

A config file is a YAML mapping with up to five sections. **`base_model` and
`dataset` are required**; `training` and `hardware` have sensible defaults;
`peft` is optional (omit it entirely for full-parameter fine-tuning).

Unknown keys anywhere are validation errors. The schema source of truth is
[`hydratune/config/schema.py`](hydratune/config/schema.py).

### `base_model`

Which model to fine-tune and how to load it.

| Field               | Type          | Default    | Notes                                                        |
| ------------------- | ------------- | ---------- | ------------------------------------------------------------ |
| `name`              | str           | *required* | Hub id (`meta-llama/Meta-Llama-3-8B`) or a local path.       |
| `revision`          | str \| null   | `null`     | Git revision (branch, tag, commit) of the Hub repo.          |
| `tokenizer_name`    | str \| null   | `null`     | Load a different tokenizer than the model's own.             |
| `trust_remote_code` | bool          | `false`    | Allow executing custom modeling code from the Hub repo.      |

### `dataset`

Where the training data lives and how its records are shaped.

| Field              | Type                  | Default             | Notes                                                                 |
| ------------------ | --------------------- | ------------------- | --------------------------------------------------------------------- |
| `path`             | str                   | *required*          | Hub dataset id, or a local `.jsonl`/`.json`/`.csv`/`.parquet` file.   |
| `format`           | enum                  | `alpaca`            | `alpaca` · `sharegpt` · `openai` · `completion`. See [Dataset formats](#dataset-formats). |
| `chat_template`    | enum \| null          | `tokenizer_default` | `tokenizer_default` · `chatml` · `llama3` · `mistral` · `zephyr` · `null`. See [Chat templates](#chat-templates). |
| `split`            | str                   | `train`             | Split to train on (Hub datasets).                                     |
| `text_field`       | str                   | `text`              | Column holding raw text; **only** used with `format: completion`.     |
| `validation_split` | float \| null         | `null`              | Fraction of training data held out for eval; `0 < x < 0.5`.           |
| `max_samples`      | int \| null           | `null`              | Cap on training records — handy for smoke tests. `≥ 1`.               |
| `shuffle`          | bool                  | `true`              | Shuffle records (seeded by `training.seed`) before training.          |

Rules enforced by the schema:

- `format: completion` requires `chat_template: null` (templates don't apply
  to raw text).
- The conversational formats (`alpaca`, `sharegpt`, `openai`) require a
  non-null `chat_template` (use `tokenizer_default` to keep the model's own).

### `training`

Optimization hyperparameters and run bookkeeping.

| Field                         | Type          | Default     | Notes                                                              |
| ----------------------------- | ------------- | ----------- | ------------------------------------------------------------------ |
| `output_dir`                  | path          | `./outputs` | Checkpoints, logs, and the final adapter/model.                    |
| `learning_rate`               | float         | `2e-4`      | `0 < x ≤ 1`.                                                       |
| `epochs`                      | float         | `1.0`       | Passes over the data; ignored when `max_steps` is set.             |
| `max_steps`                   | int \| null   | `null`      | Hard cap on optimizer steps; overrides `epochs`.                   |
| `batch_size`                  | int           | `1`         | Micro batch size **per device**.                                   |
| `gradient_accumulation_steps` | int           | `1`         | Effective batch = `batch_size × this` (per device).                |
| `warmup_steps`                | int           | `0`         | Mutually exclusive with `warmup_ratio`.                            |
| `warmup_ratio`                | float         | `0.0`       | Warmup as a fraction of total optimizer steps; `0 ≤ x ≤ 1`.        |
| `lr_scheduler`                | enum          | `cosine`    | `linear` · `cosine` · `cosine_with_restarts` · `constant` · `constant_with_warmup`. |
| `optimizer`                   | enum          | `adamw_torch` | `adamw_torch` · `adamw_torch_fused` · `adamw_8bit` · `paged_adamw_8bit` · `paged_adamw_32bit`. |
| `weight_decay`                | float         | `0.0`       | `≥ 0`.                                                             |
| `max_grad_norm`               | float         | `1.0`       | Gradient clipping; `> 0`.                                          |
| `max_seq_length`              | int           | `2048`      | Token budget per sequence after packing/truncation; `≥ 8`.         |
| `packing`                     | bool          | `true`      | Pack multiple short examples into each window.                     |
| `logging_steps`               | int           | `10`        |                                                                    |
| `save_steps`                  | int           | `500`       |                                                                    |
| `eval_steps`                  | int \| null   | `null`      | Requires `dataset.validation_split`.                               |
| `save_total_limit`            | int \| null   | `3`         | Old checkpoints beyond this are deleted.                           |
| `seed`                        | int           | `42`        | Seeds shuffling, splitting, and the trainer.                       |
| `resume_from_checkpoint`      | path \| null  | `null`      | Checkpoint directory to resume from.                               |
| `report_to`                   | enum          | `none`      | `wandb` · `tensorboard` · `none`.                                  |

> **About `warmup_ratio`:** transformers 5 removed ratio-based warmup from
> `TrainingArguments`, so HydraTune converts the ratio to a concrete step
> count itself, from the estimated total optimizer updates. With `packing`
> enabled the estimate errs slightly long (records overcount packed
> sequences) — acceptable for a warmup heuristic.

### `peft`

Parameter-efficient fine-tuning. **Omit this whole section to do
full-parameter fine-tuning.**

| Field            | Type                        | Default      | Notes                                                        |
| ---------------- | --------------------------- | ------------ | ------------------------------------------------------------ |
| `adapter`        | enum                        | *required*   | `lora` (full-precision base) or `qlora` (4-bit base).        |
| `r`              | int                         | `16`         | LoRA rank; `1–1024`.                                         |
| `lora_alpha`     | int                         | `32`         | Scaling numerator.                                           |
| `lora_dropout`   | float                       | `0.05`       | `0 ≤ x < 1`.                                                 |
| `target_modules` | list[str] \| `"all-linear"` | `all-linear` | Module names to wrap, or auto-detect all linear layers.      |
| `bias`           | enum                        | `none`       | `none` · `all` · `lora_only`.                                |
| `use_rslora`     | bool                        | `false`      | Rank-stabilized scaling (`alpha / √r`).                      |
| `quantization`   | mapping \| null             | auto         | 4-bit details; **only valid with `adapter: qlora`** (auto-filled with defaults if omitted there, rejected under `lora`). |

`quantization` sub-fields:

| Field                       | Type | Default | Notes                                             |
| --------------------------- | ---- | ------- | ------------------------------------------------- |
| `bnb_4bit_quant_type`       | enum | `nf4`   | `nf4` or `fp4`.                                   |
| `bnb_4bit_use_double_quant` | bool | `true`  | Quantize the quantization constants too.          |

The 4-bit compute dtype follows the hardware precision flags (bf16 → bfloat16,
fp16 → float16).

### `hardware`

Precision and memory/compute trade-offs.

| Field                    | Type | Default | Notes                                                          |
| ------------------------ | ---- | ------- | -------------------------------------------------------------- |
| `bf16`                   | bool | `true`  | Train in bfloat16 (Ampere or newer).                           |
| `fp16`                   | bool | `false` | Train in float16 (older GPUs). Mutually exclusive with `bf16`. |
| `flash_attention`        | bool | `false` | FlashAttention-2; requires the `flash-attn` package.           |
| `gradient_checkpointing` | bool | `true`  | Recompute activations backward to save memory.                 |
| `tf32`                   | bool | `true`  | Allow TF32 matmuls on Ampere+ (no effect elsewhere).           |

With both `bf16` and `fp16` false, training runs in float32 (useful for CPU
smoke tests; slow and memory-hungry for real models).

### Cross-section rules

Enforced at validation time, before anything loads:

- `hardware.bf16` and `hardware.fp16` cannot both be true.
- `training.warmup_steps` and `training.warmup_ratio` cannot both be set.
- `training.eval_steps` requires `dataset.validation_split`.
- `peft.adapter: qlora` requires `bf16` or `fp16` (a half-precision compute
  dtype), and is the only adapter that accepts a `quantization` block.
- `dataset.format: completion` ⇔ `dataset.chat_template: null`.

## Dataset formats

`dataset.format` declares the shape of each record. The three conversational
formats are normalized into OpenAI-style `[{role, content}, ...]` message
lists, then rendered to text through the chat template; `completion` data is
used as-is.

### `alpaca`

```json
{"instruction": "Translate to German.", "input": "Good morning!", "output": "Guten Morgen!"}
```

Required keys: `instruction`, `output`. Optional `input` is appended to the
instruction (blank line between) to form the user turn; `output` becomes the
assistant turn.

### `sharegpt`

```json
{"conversations": [
  {"from": "system", "value": "Be concise."},
  {"from": "human",  "value": "Hi!"},
  {"from": "gpt",    "value": "Hello — how can I help?"}
]}
```

Speaker tags map as `system→system`, `human`/`user→user`,
`gpt`/`assistant→assistant`. Unknown speakers are an error.

### `openai`

```json
{"messages": [
  {"role": "user", "content": "Hi!"},
  {"role": "assistant", "content": "Hello!"}
]}
```

Used verbatim; every message needs `role` and `content`.

### `completion`

```json
{"text": "Raw pretraining-style text, used exactly as written."}
```

Reads `dataset.text_field` (default `text`) from each record. No chat
template is applied — the schema requires `chat_template: null`.

### Local files vs Hub datasets

A `dataset.path` ending in `.jsonl`, `.json`, `.csv`, or `.parquet` is
treated as a local file (`.json` must be a top-level list of records);
anything else is treated as a Hub dataset id loaded with `datasets`.
`hydratune validate` can sample local `.jsonl`/`.json`/`.csv` files with no
ML dependencies installed.

## Chat templates

`dataset.chat_template` controls how message lists become token streams:

- **`tokenizer_default`** — keep whatever Jinja template the tokenizer ships.
  If the tokenizer has none, training fails fast with a clear error telling
  you to pick an explicit template.
- **`chatml`**, **`llama3`**, **`mistral`**, **`zephyr`** — overwrite
  `tokenizer.chat_template` with a known Jinja source
  ([`hydratune/data/chat_templates.py`](hydratune/data/chat_templates.py)),
  making the rendered training text deterministic regardless of base model.

Match the template to the base model family when fine-tuning an instruct
model (e.g. `llama3` for Llama-3), or pick one deliberately when teaching a
base model a chat format from scratch.

## Recipes

### QLoRA on Llama-3-8B — single 24 GB GPU

[`example_config.yaml`](example_config.yaml), in full: NF4 double-quantized
4-bit base, LoRA r=32 on all attention + MLP projections, paged 8-bit AdamW,
packing at 4096 tokens, bf16, FlashAttention-2, gradient checkpointing.

```bash
hydratune train --config example_config.yaml
```

### CPU smoke test — SmolLM2-135M, under 30 seconds

[`configs/test_smollm2.yaml`](configs/test_smollm2.yaml) fine-tunes
`HuggingFaceTB/SmolLM2-135M-Instruct` with LoRA (r=8) on a 5-row dummy
dataset in float32, so it runs anywhere:

```bash
python scripts/run_test.py
# ...
# PASS: LoRA adapter saved to output_test_adapter (34 KiB) in 6.8s
```

The script writes the dummy data, drives the real training pipeline through
the Pydantic schema, and verifies the adapter artifacts — a quick end-to-end
health check of the harness itself.

### Full fine-tune

Delete the `peft` section. Everything else stays the same; expect an order
of magnitude more memory.

### Multi-GPU

```bash
accelerate launch -m hydratune.cli train --config config.yaml
```

`batch_size` is per device; the schema's `effective_batch_size`
(`batch_size × gradient_accumulation_steps`) is also per device — multiply
by world size for the global batch.

## Errors and troubleshooting

### Config errors read like a lint report

```text
$ hydratune validate --config bad.yaml
Invalid configuration (3 error(s)):
  training.learning_rate: Input should be less than or equal to 1
  peft.lora_droput: Extra inputs are not permitted
  hardware: Value error, hardware.bf16 and hardware.fp16 are mutually exclusive
```

Every problem is listed with its `section.field` path; exit code is 1. You
never see a Python traceback for a bad config. (Field-level errors are
reported first; cross-field rules run once the fields themselves parse.)

### Out of memory

CUDA OOM during model load or training is caught and re-raised with ordered
suggestions, cheapest first:

```text
Ran out of GPU memory during training.
Suggestions:
  - Lower training.batch_size (raise gradient_accumulation_steps to keep the effective size)
  - Enable hardware.gradient_checkpointing
  - Reduce training.max_seq_length
  - Switch peft.adapter to 'qlora' to load the base model in 4-bit
  - Use a paged optimizer (training.optimizer: paged_adamw_8bit)
```

### Chat-template mismatches fail before training

If `chat_template: tokenizer_default` but the tokenizer ships no template,
you get an immediate `ChatTemplateError` naming the fix — no tokens are
processed, no GPU memory is touched.

### Dataset problems

`hydratune validate` reports per-record mismatches (`record 3: alpaca record
is missing keys ['output']`), unreadable files, and empty files. At training
time, the same conversion runs over the full dataset, so a malformed record
deep in the file still produces a `DatasetError` naming the record.

### `Training dependencies are not installed`

`hydratune train` on a base install prints exactly what to run
(`pip install "hydratune[train]"`) and exits 1.

## Architecture

```
hydratune/
├── hydratune/
│   ├── cli.py                  # Typer entry point: train / validate / export
│   ├── config/
│   │   ├── schema.py           # Pydantic v2 schema — the contract for config.yaml
│   │   └── loader.py           # YAML → HydraTuneConfig with lint-style errors
│   ├── data/
│   │   ├── formatting.py       # format registry, record → messages, cheap dataset checks
│   │   └── chat_templates.py   # named Jinja chat templates
│   ├── training/
│   │   └── trainer.py          # validated config → TRL SFTTrainer (lazy heavy imports)
│   ├── export/                 # planned: adapter merge + GGUF export
│   ├── utils/
│   │   └── errors.py           # HydraTuneError hierarchy (ConfigError, DatasetError, OOMRiskError, ...)
│   └── py.typed                # PEP 561: downstream users get our types
├── configs/test_smollm2.yaml   # CPU-friendly SmolLM2 LoRA recipe
├── scripts/run_test.py         # end-to-end smoke test
├── tests/                      # unit + integration suites (see Development)
├── example_config.yaml         # Llama-3-8B QLoRA recipe
└── pyproject.toml
```

Design principles:

- **The schema is the contract.** All invariants live in `schema.py`; the
  trainer maps an already-valid config onto library objects and never
  re-checks what the schema guarantees. Runtime checks are reserved for
  things only a live tokenizer, dataset, or GPU can reveal.
- **Heavy imports are lazy.** `torch`/`trl`/`peft` are imported inside
  functions in `trainer.py`, keeping the CLI and validation usable on
  machines without them.
- **Deliberate errors are typed.** Everything HydraTune raises on purpose
  derives from `HydraTuneError`, so the CLI catches one type, prints a
  readable message, and exits non-zero — while genuine bugs still traceback.

## Development

```bash
pip install -e ".[dev]"
pytest            # unit tests; integration tests skip without the [train] extra
ruff check .      # lint
mypy              # strict type-checking of the package
```

The test suite has two tiers:

- **Unit** (fast, no ML dependencies): schema invariants, YAML loading and
  error formatting, dataset format conversion and inspection, chat-template
  rendering, CLI behavior via Typer's test runner.
- **Integration** (`tests/test_integration.py`, needs `[train]`): builds a
  ~100k-parameter Llama-architecture checkpoint and BPE tokenizer at test
  time — no network — and drives the real `run_training` pipeline end to
  end, verifying the saved adapter artifacts, the eval-split and
  warmup-ratio paths, and the fail-fast chat-template error.

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs ruff +
strict mypy, the unit suite with coverage on Python 3.10/3.11/3.12, and the
integration suite on CPU-only torch.

Contributions welcome — please keep `pytest`, `ruff check .`, and `mypy`
green, add tests for behavior changes, and treat schema changes as API
changes (they are).

## Roadmap

- `hydratune export`: merge LoRA adapters into the base model; GGUF
  conversion for llama.cpp-compatible runtimes.
- Preference tuning (DPO/ORPO) as additional trainer backends.
- FSDP / DeepSpeed configuration passthrough in the `hardware` section.
- Weights & Biases run configuration beyond `report_to`.

## License

[Apache-2.0](LICENSE)
