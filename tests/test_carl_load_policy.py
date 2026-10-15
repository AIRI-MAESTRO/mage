"""CARL load-failure policy tests (plan item P0.3).

``_try_load_reasoning_chain`` treats ``ReasoningChain.from_json`` as a
best-effort cross-check after MAGE's own strict validation: a load failure is
non-fatal by default (warn + return ``None``) and only hard-fails when
``raise_on_error=True`` (``MAGEConfig.raise_on_carl_load_error``).

We exercise the real failure with an ``agent_skill`` chain, which CARL's
``from_json`` cannot reconstruct today (``AgentSkillStepConfig`` is missing from
the generic ``StepDescription.step_config`` union — see P0.2).
"""

from __future__ import annotations

import json
import logging

import pytest

from mmar_mage.carl_export import to_carl_compat
from mmar_mage.exceptions import MAGEValidationError
from mmar_mage.generator import _try_load_reasoning_chain
from mmar_mage.schemas import MAGEConfig

try:
    import mmar_carl  # noqa: F401

    _HAS_CARL = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_CARL = False

_needs_carl = pytest.mark.skipif(not _HAS_CARL, reason="mmar_carl not installed")


def _agent_skill_json() -> str:
    """A chain whose agent_skill step crashes CARL from_json (upstream gap)."""
    chain = {
        "name": "t",
        "steps": [
            {
                "number": 1,
                "title": "Skill",
                "step_type": "agent_skill",
                "aim": "extract",
                "dependencies": [],
                "skill": "github://anthropics/skills/skills/pdf@main",
                "skill_task": "extract tables",
            }
        ],
    }
    return json.dumps(to_carl_compat(chain))


def _tool_json() -> str:
    chain = {
        "name": "t",
        "steps": [
            {
                "number": 1,
                "title": "Fetch",
                "step_type": "tool",
                "aim": "x",
                "dependencies": [],
                "tool_name": "http_get",
            }
        ],
    }
    return json.dumps(to_carl_compat(chain))


def test_config_default_is_nonfatal():
    assert MAGEConfig().raise_on_carl_load_error is False


@_needs_carl
def test_load_failure_is_nonfatal_by_default(caplog):
    with caplog.at_level(logging.WARNING):
        result = _try_load_reasoning_chain(_agent_skill_json())
    assert result is None  # did not raise
    assert "non-fatal" in caplog.text


@_needs_carl
def test_load_failure_raises_when_strict():
    with pytest.raises(MAGEValidationError, match="from_json"):
        _try_load_reasoning_chain(_agent_skill_json(), raise_on_error=True)


@_needs_carl
def test_valid_chain_still_loads():
    rc = _try_load_reasoning_chain(_tool_json())
    assert rc is not None
    assert rc.steps[0].step_type == "tool"


def test_missing_mmar_carl_returns_none(monkeypatch):
    """If mmar_carl is absent, load is skipped (returns None), never raises."""
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "mmar_carl" or name.startswith("mmar_carl."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    # Even with raise_on_error=True, a missing package is a skip, not an error.
    assert _try_load_reasoning_chain(_tool_json(), raise_on_error=True) is None
