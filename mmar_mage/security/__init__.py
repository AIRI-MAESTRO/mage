"""Security primitives for MAGE (plan item P1.17 Phase B).

Sandboxing for synthesized tool code: a static AST analyzer that rejects banned
imports/calls, and a resource-limited subprocess runner that smoke-executes a
generated module in isolation. Used only when ``enable_tool_synthesis=True`` —
generation of executable code is opt-in, and when on, validation is mandatory.
"""

from __future__ import annotations

from .sandbox import (
    SandboxReport,
    SandboxRunResult,
    StaticViolation,
    run_module_function,
    scan_source,
    synth_inputs,
    validate_tool_source,
)

__all__ = [
    "SandboxReport",
    "SandboxRunResult",
    "StaticViolation",
    "run_module_function",
    "scan_source",
    "synth_inputs",
    "validate_tool_source",
]
