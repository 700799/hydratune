"""Exception hierarchy for HydraTune.

Every error raised deliberately by HydraTune derives from
:class:`HydraTuneError`, so the CLI can catch one type, print a readable
message, and exit non-zero without a traceback. Unexpected exceptions are
allowed to propagate so bugs stay visible.
"""

from __future__ import annotations


class HydraTuneError(Exception):
    """Base class for all errors HydraTune raises on purpose."""


class ConfigError(HydraTuneError):
    """The YAML config file is missing, unparsable, or fails schema validation."""


class DatasetError(HydraTuneError):
    """The dataset cannot be read or does not match the declared format."""


class ChatTemplateError(DatasetError):
    """The dataset format and the tokenizer/chat-template combination are incompatible."""


class ExportError(HydraTuneError):
    """A model/adapter export could not be completed."""


class OOMRiskError(HydraTuneError):
    """Training aborted because the GPU ran out of memory (or is about to)."""

    def __init__(self, message: str, *, hints: list[str] | None = None) -> None:
        self.hints = hints or []
        if self.hints:
            bullet_list = "\n".join(f"  - {hint}" for hint in self.hints)
            message = f"{message}\nSuggestions:\n{bullet_list}"
        super().__init__(message)
