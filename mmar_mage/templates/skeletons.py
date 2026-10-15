"""Built-in chain skeletons (plan item P1.3).

Each skeleton is a deterministic, domain-agnostic DAG shape used as a *soft*
seed when no stored template matches. They are intentionally short and only
suggest a step-type sequence — the StepPlanner is told to treat them as guidance,
not a rigid template (see ``generator`` template-context assembly).

A skeleton dict has:
- ``name``: stable identifier (surfaced as ``builtin:<name>``).
- ``step_type_sequence``: ordered CARL step types.
- ``abstract_dag``: minimal step dicts parallel to the sequence.
"""

from __future__ import annotations

from typing import Any


def _dag(*pairs: tuple[str, str]) -> list[dict[str, Any]]:
    """Build a simple linear abstract DAG from (step_type, title) pairs."""
    return [
        {"number": i + 1, "step_type": st, "title": title, "dependencies": [i] if i else []}
        for i, (st, title) in enumerate(pairs)
    ]


_GENERIC_TOOL_USE: dict[str, Any] = {
    "name": "generic_tool_use",
    "step_type_sequence": ["llm", "tool", "llm"],
    "abstract_dag": _dag(
        ("llm", "Interpret the request and decide which tool/action is needed"),
        ("tool", "Invoke the required tool / API with the resolved arguments"),
        ("llm", "Synthesize the tool result into the final answer"),
    ),
}

_MULTI_SOURCE_ANALYSIS: dict[str, Any] = {
    "name": "multi_source_analysis",
    "step_type_sequence": ["llm", "llm", "llm", "llm"],
    "abstract_dag": [
        {"number": 1, "step_type": "llm", "title": "Decompose into independent sub-analyses", "dependencies": []},
        {"number": 2, "step_type": "llm", "title": "Analyze the first facet", "dependencies": [1]},
        {"number": 3, "step_type": "llm", "title": "Analyze the second facet", "dependencies": [1]},
        {"number": 4, "step_type": "llm", "title": "Synthesize findings into a conclusion", "dependencies": [2, 3]},
    ],
}

_EXTRACT_TRANSFORM_EMIT: dict[str, Any] = {
    "name": "extract_transform_emit",
    "step_type_sequence": ["tool", "transform", "structured_output"],
    "abstract_dag": _dag(
        ("tool", "Fetch / extract the raw source data"),
        ("transform", "Reshape the extracted data into the working form"),
        ("structured_output", "Emit the result against the required schema"),
    ),
}

_CRITIQUE_REVISE: dict[str, Any] = {
    "name": "critique_revise",
    "step_type_sequence": ["llm", "llm", "llm"],
    "abstract_dag": _dag(
        ("llm", "Produce a first draft"),
        ("llm", "Critique the draft against the requirements"),
        ("llm", "Revise the draft using the critique"),
    ),
}

_DEFAULT: dict[str, Any] = {
    "name": "default_pipeline",
    "step_type_sequence": ["llm", "llm"],
    "abstract_dag": _dag(
        ("llm", "Analyze the request and gather what is needed"),
        ("llm", "Produce the final answer"),
    ),
}

BUILTIN_SKELETONS: dict[str, dict[str, Any]] = {
    s["name"]: s
    for s in (_GENERIC_TOOL_USE, _MULTI_SOURCE_ANALYSIS, _EXTRACT_TRANSFORM_EMIT, _CRITIQUE_REVISE, _DEFAULT)
}

# task_type keyword → skeleton, checked in order.
_SELECTORS: list[tuple[tuple[str, ...], dict[str, Any]]] = [
    (("extract", "parse", "scrape", "convert", "ingest"), _EXTRACT_TRANSFORM_EMIT),
    (("analysis", "analyze", "compare", "evaluate", "assess", "research"), _MULTI_SOURCE_ANALYSIS),
    (("tool", "agent", "api", "action", "automation", "call"), _GENERIC_TOOL_USE),
    (("generation", "generate", "write", "draft", "compose", "creation", "author"), _CRITIQUE_REVISE),
]


def select_skeleton(task_type: str, domain: str = "") -> dict[str, Any]:
    """Pick the best-fitting built-in skeleton for a task type (P1.3 fallback).

    Matches ``task_type`` (then ``domain``) against keyword groups; falls back to
    a generic two-step pipeline when nothing matches.
    """
    haystack = f"{task_type} {domain}".lower()
    for keywords, skeleton in _SELECTORS:
        if any(k in haystack for k in keywords):
            return skeleton
    return _DEFAULT
