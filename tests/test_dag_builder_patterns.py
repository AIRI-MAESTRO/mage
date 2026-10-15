"""P-4 — DAG_BUILDER_PROMPT pattern enum sourced from the topology library."""

from __future__ import annotations

from mmar_mage.prompts import DAG_BUILDER_PROMPT, build_dag_builder_prompt
from mmar_mage.topology.library import BUILTIN_TOPOLOGIES, describe_patterns_for_dag


def test_describe_patterns_lists_every_library_topology() -> None:
    rendered = describe_patterns_for_dag()
    for name in BUILTIN_TOPOLOGIES:
        assert f'"{name}"' in rendered


def test_built_prompt_fills_placeholder_with_all_patterns() -> None:
    p = build_dag_builder_prompt()
    assert "<<DAG_PATTERNS>>" not in p  # placeholder filled
    # all library patterns are now offered (was: 6 static)
    for name in BUILTIN_TOPOLOGIES:
        assert f'"{name}"' in p
    # patterns that only existed in the library (not the old static 6) now appear
    for name in ["react", "chain_of_verification", "self_consistency", "router"]:
        assert f'"{name}"' in p


def test_multi_hop_removed() -> None:
    # multi_hop was prompt-only and absent from the library; react covers it now
    assert "multi_hop" not in build_dag_builder_prompt()


def test_no_deprecated_step_types_in_examples() -> None:
    # P-1 consistency in the few-shot examples
    assert '"step_type": "eval"' not in DAG_BUILDER_PROMPT
    assert '"step_type": "aggregator"' not in DAG_BUILDER_PROMPT
