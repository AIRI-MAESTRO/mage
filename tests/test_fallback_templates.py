"""Smarter fallback paths — plan item P3.1.

When the StepPlanner is disabled, the fallback used to emit useless
``Step N / Process part N`` placeholders. It now uses a domain/task-type-keyed
template (sized to ``suggested_step_count``, output step last). The meaningful
titles + aims propagate through the DAG + description fallbacks. The
real-execution check drives a fully-disabled deep ``generate()``.
"""

from __future__ import annotations

from mmar_mage.generator import (
    _FALLBACK_GENERIC,
    _fit_template,
    _select_fallback_template,
)
from mmar_mage.schemas import DomainAnalysis


# --------------------------------------------------------------------------- #
# Template selection
# --------------------------------------------------------------------------- #
def test_selects_extraction_template():
    tmpl = _select_fallback_template(DomainAnalysis(domain="data", task_type="extraction"))
    assert tmpl[0][0] == "Parse the input"


def test_selects_comparison_template():
    tmpl = _select_fallback_template(DomainAnalysis(domain="general", task_type="comparison of options"))
    assert "criteria" in tmpl[0][0].lower()


def test_selects_by_domain_keyword_when_task_type_blank():
    tmpl = _select_fallback_template(DomainAnalysis(domain="fact verification"))
    assert "verif" in (tmpl[0][0] + tmpl[-1][0]).lower() or "check" in tmpl[1][0].lower()


def test_unknown_falls_back_to_generic():
    assert _select_fallback_template(DomainAnalysis(domain="zzz", task_type="qqq")) is _FALLBACK_GENERIC


# --------------------------------------------------------------------------- #
# _fit_template — sizing while preserving the output step
# --------------------------------------------------------------------------- #
def test_fit_exact_length():
    tmpl = [("a", "A"), ("b", "B"), ("c", "C")]
    assert _fit_template(tmpl, 3) == tmpl


def test_fit_fewer_keeps_output_step_last():
    tmpl = [("a", "A"), ("b", "B"), ("c", "C"), ("out", "OUT")]
    fitted = _fit_template(tmpl, 2)
    assert len(fitted) == 2
    assert fitted[-1] == ("out", "OUT")  # output step preserved
    assert fitted[0] == ("a", "A")


def test_fit_single_step_is_the_output():
    tmpl = [("a", "A"), ("b", "B"), ("out", "OUT")]
    assert _fit_template(tmpl, 1) == [("out", "OUT")]


def test_fit_more_pads_before_output():
    tmpl = [("a", "A"), ("out", "OUT")]
    fitted = _fit_template(tmpl, 4)
    assert len(fitted) == 4
    assert fitted[-1] == ("out", "OUT")  # output still last after padding
    assert fitted[0] == ("a", "A")


def test_fit_zero_coerced_to_one():
    assert len(_fit_template([("a", "A"), ("out", "OUT")], 0)) == 1


# --------------------------------------------------------------------------- #
# _fallback_step_plan — no more placeholders
# --------------------------------------------------------------------------- #
def test_fallback_plan_has_meaningful_titles():
    from mmar_mage.generator import MAGEGenerator

    plan = MAGEGenerator._fallback_step_plan(
        DomainAnalysis(domain="data", task_type="extraction", suggested_step_count=4)
    )
    titles = [s.title for s in plan.steps]
    assert titles == [
        "Parse the input",
        "Extract the target information",
        "Validate the extraction",
        "Emit the results",
    ]
    assert all(not t.startswith("Step ") for t in titles)
    assert all(s.rough_aim and "Process part" not in s.rough_aim for s in plan.steps)


def test_fallback_plan_respects_step_count():
    # P1.10 simplicity bias still drives the count
    from mmar_mage.generator import MAGEGenerator as G

    assert len(G._fallback_step_plan(DomainAnalysis(task_type="analysis", suggested_step_count=2)).steps) == 2
    assert len(G._fallback_step_plan(DomainAnalysis(task_type="analysis", suggested_step_count=5)).steps) == 5


# --------------------------------------------------------------------------- #
# Real-execution scenario — fully-disabled deep generate() yields a usable chain
# --------------------------------------------------------------------------- #
async def test_generate_fallback_chain_is_meaningful():
    from mmar_mage.generator import MAGEGenerator
    from mmar_mage.schemas import MAGEConfig

    config = MAGEConfig(
        mode="deep",
        api_key="test-key",
        enable_memory=False,
        enable_domain_analysis=False,  # → default DomainAnalysis (task_type="analysis" → analysis template)
        enable_step_planning=False,
        enable_dag_optimization=False,
        enable_step_descriptions=False,
        enable_topology_selection=False,
        max_retries=0,
    )
    result = await MAGEGenerator(config=config).generate("Summarize the quarterly report")

    titles = [s["title"] for s in result.chain_dict["steps"]]
    aims = [s.get("aim", "") for s in result.chain_dict["steps"]]
    assert all(not t.startswith("Step ") for t in titles)  # no placeholders
    assert all("Process part" not in a for a in aims)
    # default DomainAnalysis() has task_type="analysis" → analysis template,
    # which ends on its synthesis step.
    assert titles[-1] == "Synthesize findings"
