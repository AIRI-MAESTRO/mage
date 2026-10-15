"""Load the recommended-models preset (WP-10 / audit Part 5).

``configs/models-recommended.yaml`` captures the per-stage model picks + pricing.
This reads it into ``stage_llm_overrides`` (apply to ``MAGEConfig``) and
``pricing`` (feed :func:`mmar_mage.cost.estimate_chain_cost`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

DEFAULT_PRESET = Path(__file__).resolve().parent.parent / "configs" / "models-recommended.yaml"


def _pricing_per_1k(per_m: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """``{model: [in_per_M, out_per_M]}`` → ``{model: (in_per_1k, out_per_1k)}`` for cost.py."""
    out: dict[str, tuple[float, float]] = {}
    for model, v in (per_m or {}).items():
        if isinstance(v, list | tuple) and len(v) >= 2:
            out[model] = (float(v[0]) / 1000.0, float(v[1]) / 1000.0)
    return out


def load_models_preset(path: str | Path | None = None) -> dict[str, Any]:
    """Return ``{"stage_llm_overrides": {...}, "pricing": {model: (in_1k, out_1k)}}``.

    ``stage_llm_overrides`` is ready to pass to ``MAGEConfig(stage_llm_overrides=...)``
    (bare dicts are promoted to ``StageLLMOverride``); ``pricing`` is the per-1k
    mapping ``estimate_chain_cost`` expects.
    """
    p = Path(path) if path else DEFAULT_PRESET
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {
        "stage_llm_overrides": data.get("stage_llm_overrides", {}) or {},
        "pricing": _pricing_per_1k(data.get("pricing_per_million_usd", {})),
    }
