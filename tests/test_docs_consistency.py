"""T40 — README + ARCHITECTURE docs list the real deep-mode pipeline stages."""

from __future__ import annotations

from pathlib import Path

# Deep-mode stage agents the audit flagged (README claimed 5 stages; the code has
# 6+, incl. TopologySelector + the capability merge). Each must be named in the
# user-facing docs so the pipeline description can't silently drift again.
_CORE_STAGES = ["DomainAnalyzer", "StepPlanner", "TopologySelector", "DAGBuilder", "StepDescriber", "CapabilityLookup"]


def test_readme_lists_core_pipeline_stages() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    for stage in _CORE_STAGES:
        assert stage in readme, f"README.md doesn't mention the {stage} stage"


def test_architecture_lists_core_pipeline_stages() -> None:
    arch = Path("docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    for stage in _CORE_STAGES:
        assert stage in arch, f"docs/ARCHITECTURE.md doesn't mention the {stage} stage"


def test_documented_stage_agents_exist() -> None:
    # the docs claim TopologySelector + CapabilityLookup stages — assert the agent
    # modules really exist (catches a stage removed from code but left in the docs)
    assert Path("mmar_mage/agents/topology_selector.py").exists()
    assert Path("mmar_mage/agents/capability_lookup_agent.py").exists()


def test_incident_summary_in_release_notes() -> None:
    notes = Path("docs/RELEASE_NOTES.md").read_text(encoding="utf-8")
    assert "Incident in one paragraph" in notes
    assert "Invalid IDNA hostname" in notes  # the concrete run-096 symptom


def test_care_side_fixes_spec_covers_care_tasks() -> None:
    # T6/T7/T14/T20 are CARE-side — spec'd here since CARE isn't in this working copy.
    spec = Path("docs/care-side-fixes.md").read_text(encoding="utf-8")
    assert "T6 — Tool errors must be errors" in spec
    assert "find_masked_tool_errors" in spec  # references the shipped MAGE-side detector (T34)
    assert "T7 — Synthesiser must own failures" in spec
    assert "never tell the user to rephrase" in spec  # the core T7 rule
    assert "T14 — Token + cost accounting" in spec
    assert "cost_usd_for_usage" in spec  # references the shipped MAGE-side cost helper
    assert "T20 — Honest run status" in spec
    assert "on_error" in spec  # T20 mirrors the MAGEProgressCallback.on_error pattern
