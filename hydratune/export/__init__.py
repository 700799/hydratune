"""Model export: merging LoRA/QLoRA adapters into their base model.

GGUF conversion is not implemented yet — it depends on llama.cpp tooling that
is not a pip dependency.
"""

from hydratune.export.merge import merge_adapter

__all__ = ["merge_adapter"]
