"""Tests for the NL-driven chain-edit mode.

Covers the three layers added for editing existing chains:

* :mod:`mmar_mage.chain_edit` — the deterministic apply layer (no LLM).
* :class:`mmar_mage.agents.chain_editor.ChainEditPlanner` — NL → edit plan
  (LLM mocked at ``generate_json``).
* :meth:`mmar_mage.generator.MAGEGenerator.edit` — resolve → plan → apply →
  validate → optional save (memory + LLM faked).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from mmar_mage import apply_edit_plan
from mmar_mage.agents.chain_editor import ChainEditPlanner
from mmar_mage.chain_edit import apply_edit
from mmar_mage.exceptions import MAGEValidationError
from mmar_mage.generator import MAGEGenerator
from mmar_mage.schemas import ChainEdit, ChainEditPlan, MAGEConfig, MAGEEditResult


def _sample_chain() -> dict[str, Any]:
    """A small, strict-CARL-valid 3-step chain (survives validate_or_repair)."""
    return {
        "name": "Doc Analysis",
        "max_workers": 2,
        "timeout": 300.0,
        "search_config": {
            "strategy": "substring",
            "substring_config": {
                "case_sensitive": False,
                "min_word_length": 3,
                "max_matches_per_query": 5,
            },
        },
        "steps": [
            {
                "step_type": "llm", "number": 1, "title": "Classify",
                "aim": "classify doc", "reasoning_questions": "q",
                "step_context_queries": ["x"], "stage_action": "do",
                "example_reasoning": "e", "dependencies": [], "retry_max": 3,
                "llm_config": None,
            },
            {
                "step_type": "llm", "number": 2, "title": "Extract",
                "aim": "extract facts", "reasoning_questions": "q",
                "step_context_queries": ["x"], "stage_action": "do",
                "example_reasoning": "e", "dependencies": [1], "retry_max": 3,
                "llm_config": None,
            },
            {
                "step_type": "llm", "number": 3, "title": "Synthesize",
                "aim": "synthesize", "reasoning_questions": "q",
                "step_context_queries": ["x"], "stage_action": "do",
                "example_reasoning": "e", "dependencies": [1, 2], "retry_max": 3,
                "llm_config": None,
            },
        ],
    }


# ---------------------------------------------------------------------------
# apply_edit / apply_edit_plan — deterministic, no LLM
# ---------------------------------------------------------------------------


class TestApplyEdit:
    def test_edit_field_patches_step(self) -> None:
        out = apply_edit_plan(
            _sample_chain(),
            [ChainEdit(op="edit_field", target_step_number=2, fields={"aim": "extract tables too"})],
        )
        assert out["steps"][1]["aim"] == "extract tables too"
        assert out["steps"][0]["aim"] == "classify doc"  # untouched

    def test_edit_field_never_corrupts_number(self) -> None:
        out = apply_edit_plan(
            _sample_chain(),
            [ChainEdit(op="edit_field", target_step_number=1, fields={"number": 99, "aim": "x"})],
        )
        assert out["steps"][0]["number"] == 1  # number is owned by the renumberer

    def test_edit_field_deep_merges_step_config(self) -> None:
        """Regression: patching a nested step_config key must keep siblings.

        A tool step's step_config carries {tool_name, input_mapping}. Patching
        only input_mapping must NOT drop tool_name (else ToolStepConfig 400)."""
        chain = {
            "name": "c",
            "steps": [
                {
                    "step_type": "tool",
                    "number": 1,
                    "title": "Search",
                    "dependencies": [],
                    "step_config": {
                        "tool_name": "web_search",
                        "input_mapping": {"query": "$history[1]"},
                    },
                }
            ],
        }
        out = apply_edit_plan(
            chain,
            [
                ChainEdit(
                    op="edit_field",
                    target_step_number=1,
                    fields={"step_config": {"input_mapping": {"query": "$outer_context"}}},
                )
            ],
        )
        sc = out["steps"][0]["step_config"]
        assert sc["tool_name"] == "web_search"  # sibling preserved (was dropped before)
        assert sc["input_mapping"]["query"] == "$outer_context"  # patch applied

    def test_set_chain_field(self) -> None:
        out = apply_edit_plan(
            _sample_chain(),
            [ChainEdit(op="set_chain_field", fields={"name": "Renamed", "max_workers": 4})],
        )
        assert out["name"] == "Renamed"
        assert out["max_workers"] == 4

    def test_set_chain_field_ignores_unlisted_keys(self) -> None:
        out = apply_edit_plan(
            _sample_chain(), [ChainEdit(op="set_chain_field", fields={"steps": []})]
        )
        assert len(out["steps"]) == 3  # cannot wipe steps via set_chain_field

    def test_set_dependencies(self) -> None:
        out = apply_edit_plan(
            _sample_chain(), [ChainEdit(op="set_dependencies", target_step_number=3, value=[1])]
        )
        s3 = next(s for s in out["steps"] if s["number"] == 3)
        assert s3["dependencies"] == [1]

    def test_insert_step_grows_and_renumbers(self) -> None:
        new = {
            "step_type": "llm", "title": "Validate", "aim": "validate",
            "reasoning_questions": "q", "step_context_queries": ["x"],
            "stage_action": "do", "dependencies": [3],
        }
        out = apply_edit_plan(
            _sample_chain(), [ChainEdit(op="insert_step", target_step_number=3, new_step=new)]
        )
        assert len(out["steps"]) == 4
        assert [s["number"] for s in out["steps"]] == [1, 2, 3, 4]

    def test_remove_step_shrinks_and_rewires(self) -> None:
        out = apply_edit_plan(_sample_chain(), [ChainEdit(op="remove_step", target_step_number=2)])
        nums = [s["number"] for s in out["steps"]]
        assert nums == [1, 2]
        for s in out["steps"]:  # no dangling dependency on the removed step
            assert all(d in nums for d in s.get("dependencies", []))

    def test_remove_step_preserves_and_remaps_surviving_deps(self) -> None:
        # Regression: removing step 1 must KEEP C's dependency on B and remap it
        # (B old#2 → new#1). A bug in _renumber dropped a valid dep when its old
        # number collided with the step's new number.
        chain = {
            "name": "t",
            "steps": [
                {"step_type": "llm", "number": 1, "title": "A", "aim": "a", "dependencies": []},
                {"step_type": "llm", "number": 2, "title": "B", "aim": "b", "dependencies": [1]},
                {"step_type": "llm", "number": 3, "title": "C", "aim": "c", "dependencies": [1, 2]},
            ],
        }
        out = apply_edit_plan(chain, [ChainEdit(op="remove_step", target_step_number=1)])
        by_title = {s["title"]: s for s in out["steps"]}
        assert by_title["B"]["dependencies"] == []  # only depended on the removed A
        assert by_title["C"]["dependencies"] == [by_title["B"]["number"]]  # still on B, remapped

    def test_change_step_type(self) -> None:
        out = apply_edit_plan(
            _sample_chain(),
            [ChainEdit(op="change_step_type", target_step_number=1, new_step={"step_type": "structured_output"})],
        )
        assert out["steps"][0]["step_type"] == "structured_output"

    def test_field_edits_apply_before_structural(self) -> None:
        # field edit on step 3 + remove step 1 → the field must land on the
        # ORIGINAL step 3 (field edits run first, before renumbering).
        out = apply_edit_plan(
            _sample_chain(),
            [
                ChainEdit(op="edit_field", target_step_number=3, fields={"aim": "FINAL"}),
                ChainEdit(op="remove_step", target_step_number=1),
            ],
        )
        assert len(out["steps"]) == 2
        assert any(s["aim"] == "FINAL" for s in out["steps"])

    def test_unknown_op_is_noop(self) -> None:
        chain = _sample_chain()
        fake = SimpleNamespace(
            op="bogus", target_step_number=1, new_step=None, fields=None, value=None
        )
        assert apply_edit(chain, fake) == chain  # type: ignore[arg-type]

    def test_empty_plan_is_noop(self) -> None:
        chain = _sample_chain()
        assert apply_edit_plan(chain, []) == chain

    def test_does_not_mutate_input(self) -> None:
        chain = _sample_chain()
        apply_edit_plan(chain, [ChainEdit(op="edit_field", target_step_number=1, fields={"aim": "z"})])
        assert chain["steps"][0]["aim"] == "classify doc"


# ---------------------------------------------------------------------------
# ChainEditPlanner — mock the LLM's generate_json
# ---------------------------------------------------------------------------


def _fake_llm(return_value: Any = None, side_effect: Any = None) -> Any:
    llm = SimpleNamespace()
    llm.generate_json = AsyncMock(return_value=return_value, side_effect=side_effect)
    return llm


class TestChainEditPlanner:
    async def test_parses_plan(self) -> None:
        llm = _fake_llm(
            {"summary": "did it", "edits": [
                {"op": "edit_field", "target_step_number": 2, "fields": {"aim": "x"}, "rationale": "r"}
            ]}
        )
        plan = await ChainEditPlanner(llm).plan(_sample_chain(), "make step 2 do x")
        assert isinstance(plan, ChainEditPlan)
        assert plan.summary == "did it"
        assert len(plan.edits) == 1 and plan.edits[0].op == "edit_field"

    async def test_drops_edit_for_missing_step(self) -> None:
        llm = _fake_llm(
            {"summary": "", "edits": [
                {"op": "edit_field", "target_step_number": 99, "fields": {"aim": "x"}},
                {"op": "edit_field", "target_step_number": 1, "fields": {"aim": "y"}},
            ]}
        )
        plan = await ChainEditPlanner(llm).plan(_sample_chain(), "...")
        assert [e.target_step_number for e in plan.edits] == [1]

    async def test_llm_failure_returns_empty_plan(self) -> None:
        llm = _fake_llm(side_effect=RuntimeError("boom"))
        plan = await ChainEditPlanner(llm).plan(_sample_chain(), "...")
        assert plan.edits == [] and plan.summary == ""

    async def test_malformed_edits_dropped(self) -> None:
        llm = _fake_llm(
            {"summary": "s", "edits": [
                "not a dict",
                {"op": "bogus_op"},
                {"op": "set_chain_field", "fields": {"name": "N"}},
            ]}
        )
        plan = await ChainEditPlanner(llm).plan(_sample_chain(), "...")
        assert [e.op for e in plan.edits] == ["set_chain_field"]


# ---------------------------------------------------------------------------
# MAGEGenerator.edit — resolve → plan → apply → validate → save
# ---------------------------------------------------------------------------


class FakeMemory:
    """Stand-in for MemoryManager: records save calls, serves canned reads."""

    def __init__(
        self,
        *,
        chain: dict[str, Any] | None = None,
        hits: list[dict[str, Any]] | None = None,
        saved_id: str = "newver1",
    ) -> None:
        self._chain = chain
        self._hits = hits or []
        self.saved_id = saved_id
        self.save_calls: list[dict[str, Any]] = []

    async def load_chain(self, entity_id: str, *, channel: str = "latest") -> dict[str, Any] | None:
        return self._chain

    async def search_similar(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        return list(self._hits)

    async def save_chain(
        self,
        chain_dict: dict[str, Any],
        *,
        domain: str = "general",
        name: str | None = None,
        entity_id: str | None = None,
        channel: str = "latest",
        **_: Any,
    ) -> str:
        self.save_calls.append({"entity_id": entity_id, "channel": channel, "name": name})
        return self.saved_id


def _make_generator(*, plan: dict[str, Any], memory: FakeMemory) -> MAGEGenerator:
    gen = MAGEGenerator(config=MAGEConfig(model="gpt-4o", api_key="test", enable_memory=True))
    gen._memory = memory  # type: ignore[assignment]
    gen._llm.generate_json = AsyncMock(return_value=plan)  # type: ignore[method-assign]
    gen._llm.resolve_model = AsyncMock(return_value=None)  # type: ignore[method-assign]
    return gen


class TestEdit:
    async def test_edit_by_id_applies_and_saves_new_version(self) -> None:
        mem = FakeMemory(chain=_sample_chain(), saved_id="v2")
        plan = {"summary": "renamed", "edits": [
            {"op": "set_chain_field", "fields": {"name": "Invoice QA"}, "rationale": "r"}
        ]}
        gen = _make_generator(plan=plan, memory=mem)
        res = await gen.edit("rename to Invoice QA", entity_id="abc", save=True)
        assert isinstance(res, MAGEEditResult)
        assert res.resolved_by == "id"
        assert res.chain_dict["name"] == "Invoice QA"
        assert res.summary == "renamed"
        assert res.saved is True and res.memory_key == "v2"
        assert mem.save_calls and mem.save_calls[0]["entity_id"] == "abc"  # new version, same id

    async def test_edit_no_save_skips_persist(self) -> None:
        mem = FakeMemory(chain=_sample_chain())
        plan = {"summary": "s", "edits": [
            {"op": "edit_field", "target_step_number": 1, "fields": {"aim": "z"}}
        ]}
        gen = _make_generator(plan=plan, memory=mem)
        res = await gen.edit("...", entity_id="abc", save=False)
        assert res.saved is False and res.memory_key is None
        assert mem.save_calls == []

    async def test_edit_provided_chain(self) -> None:
        mem = FakeMemory()
        gen = _make_generator(plan={"summary": "", "edits": []}, memory=mem)
        res = await gen.edit("no-op", chain=_sample_chain())
        assert res.resolved_by == "provided"
        assert res.chain_dict["name"] == "Doc Analysis"
        assert res.before_chain_dict["name"] == "Doc Analysis"

    async def test_edit_missing_id_raises(self) -> None:
        mem = FakeMemory(chain=None)  # load_chain returns None
        gen = _make_generator(plan={"summary": "", "edits": []}, memory=mem)
        with pytest.raises(MAGEValidationError):
            await gen.edit("...", entity_id="missing")

    async def test_resolve_single_hit_by_search(self) -> None:
        mem = FakeMemory(chain=_sample_chain(), hits=[{"entity_id": "only", "name": "X", "score": 0.9}])
        plan = {"summary": "s", "edits": [
            {"op": "edit_field", "target_step_number": 1, "fields": {"aim": "z"}}
        ]}
        gen = _make_generator(plan=plan, memory=mem)
        res = await gen.edit("change the doc analyzer")
        assert res.resolved_by == "search"
        assert res.entity_id == "only"
        assert res.needs_disambiguation is False

    async def test_resolve_multi_hit_disambiguation(self) -> None:
        mem = FakeMemory(hits=[
            {"entity_id": "a", "name": "A", "score": 0.8},
            {"entity_id": "b", "name": "B", "score": 0.7},
        ])
        gen = _make_generator(plan={"summary": "", "edits": []}, memory=mem)
        res = await gen.edit("change something")
        assert res.needs_disambiguation is True
        assert {c.entity_id for c in res.candidates} == {"a", "b"}
        assert res.chain_dict == {}  # no edit performed
        assert mem.save_calls == []

    async def test_resolve_zero_hits_raises(self) -> None:
        mem = FakeMemory(hits=[])
        gen = _make_generator(plan={"summary": "", "edits": []}, memory=mem)
        with pytest.raises(MAGEValidationError):
            await gen.edit("nonexistent thing")

    async def test_edit_applies_multi_op_plan(self) -> None:
        mem = FakeMemory(chain=_sample_chain())
        plan = {"summary": "rename + add step", "edits": [
            {"op": "set_chain_field", "fields": {"name": "New Name"}, "rationale": "r"},
            {"op": "edit_field", "target_step_number": 2, "fields": {"aim": "extract tables"}, "rationale": "r"},
        ]}
        gen = _make_generator(plan=plan, memory=mem)
        res = await gen.edit("rename and tweak step 2", entity_id="abc")
        assert res.chain_dict["name"] == "New Name"
        assert any(s["aim"] == "extract tables" for s in res.chain_dict["steps"])
        assert len(res.edits) == 2
