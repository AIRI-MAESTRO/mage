"""Built-in chain skeletons used as a synthesis fallback (plan item P1.3).

When ``TemplateAssemblerAgent`` finds no stored template for a query, it emits
one of these deterministic skeletons as *soft* guidance for the StepPlanner
(``abstract_dag`` + ``step_type_sequence``), keyed off the task type.
"""

from __future__ import annotations

from .skeletons import BUILTIN_SKELETONS, select_skeleton

__all__ = ["BUILTIN_SKELETONS", "select_skeleton"]
