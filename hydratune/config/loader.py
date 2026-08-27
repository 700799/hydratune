"""Load and validate a HydraTune YAML config file.

Translates YAML parse errors and Pydantic validation errors into a single
:class:`~hydratune.utils.errors.ConfigError` whose message reads like a
lint report (``section.field: problem``), so the CLI never shows users a
raw traceback for a bad config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from hydratune.config.schema import HydraTuneConfig
from hydratune.utils.errors import ConfigError


def _format_validation_error(exc: ValidationError) -> str:
    lines = [f"Invalid configuration ({exc.error_count()} error(s)):"]
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  {location}: {error['msg']}")
    return "\n".join(lines)


def load_config(path: str | Path) -> HydraTuneConfig:
    """Parse ``path`` as YAML and validate it against :class:`HydraTuneConfig`.

    Raises:
        ConfigError: if the file is missing, is not valid YAML, is not a
            mapping at the top level, or fails schema validation.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file is not valid YAML: {path}\n{exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Config file could not be read: {path}\n{exc}") from exc

    if raw is None:
        raise ConfigError(f"Config file is empty: {path}")
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Top level of {path} must be a mapping of config sections, "
            f"got {type(raw).__name__}"
        )

    try:
        return HydraTuneConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc
