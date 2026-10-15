"""Built-in chain-skeleton library + selector (plan item P1.3)."""

from __future__ import annotations

from mmar_mage.templates import BUILTIN_SKELETONS, select_skeleton


def test_every_skeleton_is_well_formed():
    for name, skel in BUILTIN_SKELETONS.items():
        assert skel["name"] == name
        seq = skel["step_type_sequence"]
        dag = skel["abstract_dag"]
        assert seq and dag, name
        # the DAG is parallel to the sequence and numbered 1..N
        assert len(dag) == len(seq), name
        assert [s["step_type"] for s in dag] == seq, name
        assert [s["number"] for s in dag] == list(range(1, len(dag) + 1)), name


def test_select_extraction_task():
    assert select_skeleton("data extraction").get("name") == "extract_transform_emit"
    assert select_skeleton("parse logs").get("name") == "extract_transform_emit"


def test_select_analysis_task():
    assert select_skeleton("analysis").get("name") == "multi_source_analysis"
    assert select_skeleton("compare vendors").get("name") == "multi_source_analysis"


def test_select_tool_use_task():
    assert select_skeleton("tool calling").get("name") == "generic_tool_use"
    assert select_skeleton("api automation").get("name") == "generic_tool_use"


def test_select_generation_task():
    assert select_skeleton("content generation").get("name") == "critique_revise"
    assert select_skeleton("write a report").get("name") == "critique_revise"


def test_select_falls_back_to_default():
    assert select_skeleton("").get("name") == "default_pipeline"
    assert select_skeleton("something unmappable").get("name") == "default_pipeline"


def test_domain_can_drive_selection_when_task_type_is_vague():
    # task_type empty, but domain carries the signal
    assert select_skeleton("", domain="document extraction pipeline").get("name") == "extract_transform_emit"
