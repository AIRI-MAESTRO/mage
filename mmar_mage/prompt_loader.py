"""YAML-backed prompt loader (§8 P2).

All 26 prompt strings used by the MAGE pipeline live as keys of a
single YAML file under ``mmar_mage/prompts_data/prompts.yaml``. Power
users (e.g. CARE's settings screen) can customise prompts per-
deployment by:

1. Editing the bundled ``prompts.yaml`` in their install.
2. Pointing ``MAGE_PROMPTS_PATH`` env var at a custom YAML file.
3. Calling ``load_prompts(path)`` directly with a ``pathlib.Path``.

``mmar_mage/prompts.py`` calls :func:`load_prompts` at import time
and re-exposes every value as a module-level constant
(``STEP_PLANNER_PROMPT``, ``DAG_BUILDER_PROMPT``, ...), so existing
agents that do ``from .prompts import X`` keep working unchanged.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


_DEFAULT_YAML_PATH = Path(__file__).parent / "prompts_data" / "prompts.yaml"

_ENV_PROMPTS_PATH = "MAGE_PROMPTS_PATH"


def default_prompts_path() -> Path:
    """Return the bundled prompts YAML path.

    Respects the ``MAGE_PROMPTS_PATH`` env var when set so users can
    swap in a custom YAML without touching code.
    """
    override = os.environ.get(_ENV_PROMPTS_PATH)
    if override:
        return Path(override)
    return _DEFAULT_YAML_PATH


def load_prompts(path: Path | str | None = None) -> dict[str, str]:
    """Load every prompt from a YAML file.

    Parameters
    ----------
    path : Path | str | None
        Where to read the YAML from. ``None`` resolves to
        :func:`default_prompts_path` (env-var-aware).

    Returns
    -------
    dict[str, str]
        Mapping of prompt name → prompt body. Every value is a
        non-empty string; missing or non-string values raise
        :class:`ValueError`.

    Raises
    ------
    FileNotFoundError
        If the YAML file doesn't exist on disk.
    ValueError
        If the YAML loads as something other than a string-keyed,
        string-valued mapping.
    """
    resolved = Path(path) if path is not None else default_prompts_path()
    if not resolved.is_file():
        raise FileNotFoundError(f"prompt YAML not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as fh:
        data: Any = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"prompt YAML at {resolved} must contain a top-level mapping, got {type(data).__name__}")

    out: dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError(f"prompt YAML key {key!r} is not a string")
        if not isinstance(value, str):
            raise ValueError(f"prompt YAML value for {key!r} is not a string (got {type(value).__name__})")
        if not value.strip():
            raise ValueError(f"prompt YAML value for {key!r} is empty")
        out[key] = value

    logger.debug("Loaded %d prompts from %s", len(out), resolved)
    return out
