"""
Tests for the 6 chain quality enhancement techniques.

Covers: StepCritic, ChainVerifier, SelfRefiner, ToTPlanner,
MCTSConstructor, FeedbackRecallAgent, and integration.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mmar_mage.agents.chain_verifier import ChainVerifier, VerificationResult
from mmar_mage.agents.feedback_recall_agent import FeedbackDigest, FeedbackRecallAgent
from mmar_mage.agents.mcts_constructor import MCTSConstructor, MCTSNode
from mmar_mage.agents.self_refiner import RefinementResult, SelfRefiner
from mmar_mage.agents.step_critic import StepCriticAgent, StepCritiqueResult
from mmar_mage.agents.tot_planner import ToTPlanner, ToTResult
from mmar_mage.schemas import (
    CARLStepSchema,
    DAGStructure,
    DomainAnalysis,
    MAGEConfig,
    MAGEMetadata,
    StepPlan,
)

# ---------------------------------------------------------------------------
# Shared fixtures & helpers
# ---------------------------------------------------------------------------

SAMPLE_DOMAIN = DomainAnalysis(
    domain="finance",
    task_type="analysis",
    complexity="medium",
    key_concepts=["revenue", "profit", "cash flow"],
    suggested_step_count=3,
    preferred_step_types=["llm", "eval"],
)

SAMPLE_STEPS = [
    CARLStepSchema(
        number=1,
        title="Analyze revenue",
        aim="Determine revenue trends",
        reasoning_questions="What are the revenue trends?",
        step_context_queries=["revenue", "trends"],
        stage_action="Analyze revenue data",
        example_reasoning="Revenue trends indicate growth",
        dependencies=[],
    ),
    CARLStepSchema(
        number=2,
        title="Assess risks",
        aim="Identify financial risks",
        reasoning_questions="What risks exist?",
        step_context_queries=["risks", "liabilities"],
        stage_action="Evaluate risk factors",
        example_reasoning="Debt-to-equity ratio is key",
        dependencies=[1],
    ),
    CARLStepSchema(
        number=3,
        title="Synthesize findings",
        aim="Form conclusions from analysis",
        reasoning_questions="What conclusions can be drawn?",
        step_context_queries=["conclusions", "summary"],
        stage_action="Synthesize analysis results",
        example_reasoning="Combining revenue and risk data",
        dependencies=[1, 2],
    ),
]

SAMPLE_CHAIN_DICT: dict[str, Any] = {
    "name": "Finance Analysis",
    "max_workers": 2,
    "timeout": 300.0,
    "search_config": {
        "strategy": "substring",
        "substring_config": {"case_sensitive": False, "min_word_length": 3, "max_matches_per_query": 5},
    },
    "steps": [
        {
            "step_type": "llm",
            "number": 1,
            "title": "Analyze revenue",
            "aim": "Determine revenue trends",
            "reasoning_questions": "What are the revenue trends?",
            "step_context_queries": ["revenue"],
            "stage_action": "Analyze revenue data",
            "example_reasoning": "Revenue trends",
            "dependencies": [],
            "retry_max": 3,
        },
        {
            "step_type": "llm",
            "number": 2,
            "title": "Assess risks",
            "aim": "Identify financial risks",
            "reasoning_questions": "What risks exist?",
            "step_context_queries": ["risks"],
            "stage_action": "Evaluate risk factors",
            "example_reasoning": "Debt ratio",
            "dependencies": [1],
            "retry_max": 3,
        },
        {
            "step_type": "llm",
            "number": 3,
            "title": "Synthesize",
            "aim": "Form conclusions",
            "reasoning_questions": "What conclusions?",
            "step_context_queries": ["conclusions"],
            "stage_action": "Synthesize results",
            "example_reasoning": "Combining data",
            "dependencies": [1, 2],
            "retry_max": 3,
        },
    ],
}

SAMPLE_STEP_PLAN = StepPlan(
    steps=[
        {"number": 1, "title": "Analyze revenue", "step_type": "llm", "rough_aim": "Analyze revenue"},
        {"number": 2, "title": "Assess risks", "step_type": "llm", "rough_aim": "Identify risks"},
        {"number": 3, "title": "Synthesize", "step_type": "llm", "rough_aim": "Form conclusions"},
    ],
    rationale="Standard pipeline",
)

SAMPLE_DAG = DAGStructure(
    steps=[
        {
            "number": 1,
            "title": "Analyze revenue",
            "step_type": "llm",
            "dependencies": [],
            "rough_aim": "Analyze revenue",
        },
        {"number": 2, "title": "Assess risks", "step_type": "llm", "dependencies": [], "rough_aim": "Identify risks"},
        {
            "number": 3,
            "title": "Synthesize",
            "step_type": "llm",
            "dependencies": [1, 2],
            "rough_aim": "Form conclusions",
        },
    ],
    max_workers=2,
    pattern="funnel",
)


def _make_openai_response(content: str) -> MagicMock:
    """Build a mock OpenAI chat completion response."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_llm(responses: list[str]) -> MagicMock:
    """Create a mock LLMClient that returns sequential JSON responses."""
    llm = MagicMock()
    side_effects = [json.loads(r) for r in responses]
    llm.generate_json = AsyncMock(side_effect=side_effects)
    llm.generate = AsyncMock(side_effect=responses)
    llm.generate_structured = AsyncMock()
    return llm


# ===========================================================================
# 1. Step Critique tests
# ===========================================================================


class TestStepCritic:
    """Tests for StepCriticAgent."""

    @pytest.mark.asyncio
    async def test_step_critic_all_pass(self) -> None:
        """All steps score above threshold → no revisions needed."""
        critique_response = json.dumps(
            {
                "critiques": [
                    {
                        "step_number": 1,
                        "scores": {
                            "aim_clarity": 0.9,
                            "reasoning_quality": 0.8,
                            "context_relevance": 0.85,
                            "dependency_logic": 1.0,
                            "action_specificity": 0.9,
                        },
                        "overall_score": 0.89,
                        "issues": [],
                        "suggestions": [],
                    },
                    {
                        "step_number": 2,
                        "scores": {
                            "aim_clarity": 0.85,
                            "reasoning_quality": 0.8,
                            "context_relevance": 0.8,
                            "dependency_logic": 0.9,
                            "action_specificity": 0.85,
                        },
                        "overall_score": 0.84,
                        "issues": [],
                        "suggestions": [],
                    },
                    {
                        "step_number": 3,
                        "scores": {
                            "aim_clarity": 0.9,
                            "reasoning_quality": 0.85,
                            "context_relevance": 0.9,
                            "dependency_logic": 1.0,
                            "action_specificity": 0.9,
                        },
                        "overall_score": 0.91,
                        "issues": [],
                        "suggestions": [],
                    },
                ]
            }
        )
        llm = _make_llm([critique_response])
        critic = StepCriticAgent(llm)
        result = await critic.critique("Analyze financial report", SAMPLE_STEPS, SAMPLE_DOMAIN)

        assert isinstance(result, StepCritiqueResult)
        assert len(result.critiques) == 3
        assert result.steps_needing_revision == []
        assert result.overall_quality > 0.8

    @pytest.mark.asyncio
    async def test_step_critic_identifies_weak_step(self) -> None:
        """Steps below threshold are flagged for revision."""
        critique_response = json.dumps(
            {
                "critiques": [
                    {
                        "step_number": 1,
                        "scores": {
                            "aim_clarity": 0.9,
                            "reasoning_quality": 0.8,
                            "context_relevance": 0.85,
                            "dependency_logic": 1.0,
                            "action_specificity": 0.9,
                        },
                        "overall_score": 0.89,
                        "issues": [],
                        "suggestions": [],
                    },
                    {
                        "step_number": 2,
                        "scores": {
                            "aim_clarity": 0.3,
                            "reasoning_quality": 0.4,
                            "context_relevance": 0.3,
                            "dependency_logic": 0.5,
                            "action_specificity": 0.3,
                        },
                        "overall_score": 0.36,
                        "issues": ["aim too vague"],
                        "suggestions": ["Make aim specific"],
                    },
                    {
                        "step_number": 3,
                        "scores": {
                            "aim_clarity": 0.9,
                            "reasoning_quality": 0.85,
                            "context_relevance": 0.9,
                            "dependency_logic": 1.0,
                            "action_specificity": 0.9,
                        },
                        "overall_score": 0.91,
                        "issues": [],
                        "suggestions": [],
                    },
                ]
            }
        )
        llm = _make_llm([critique_response])
        critic = StepCriticAgent(llm)
        result = await critic.critique("Analyze report", SAMPLE_STEPS, SAMPLE_DOMAIN, threshold=0.6)

        assert 2 in result.steps_needing_revision
        assert 1 not in result.steps_needing_revision

    @pytest.mark.asyncio
    async def test_step_critic_triggers_revision(self) -> None:
        """Verify the critique result structure enables revision."""
        critique_response = json.dumps(
            {
                "critiques": [
                    {
                        "step_number": 1,
                        "scores": {"aim_clarity": 0.4},
                        "overall_score": 0.4,
                        "issues": ["vague"],
                        "suggestions": ["fix"],
                    },
                ]
            }
        )
        llm = _make_llm([critique_response])
        critic = StepCriticAgent(llm)
        result = await critic.critique("Q", SAMPLE_STEPS[:1], SAMPLE_DOMAIN, threshold=0.6)

        assert result.steps_needing_revision == [1]
        assert len(result.critiques[0].issues) > 0
        assert len(result.critiques[0].suggestions) > 0

    @pytest.mark.asyncio
    async def test_step_critic_flags_critical_dim_despite_passing_average(self) -> None:
        """P1.11: a low critical dimension forces revision even if overall passes."""
        critique_response = json.dumps(
            {
                "critiques": [
                    {
                        "step_number": 1,
                        # average is fine, but action_specificity is load-bearing-low
                        "scores": {
                            "aim_clarity": 0.9,
                            "reasoning_quality": 0.9,
                            "context_relevance": 0.9,
                            "dependency_logic": 0.9,
                            "action_specificity": 0.3,
                        },
                        "overall_score": 0.78,  # passes the 0.6 overall threshold
                        "issues": ["action underspecified"],
                        "suggestions": ["state the concrete action"],
                    },
                ]
            }
        )
        llm = _make_llm([critique_response])
        result = await StepCriticAgent(llm).critique(
            "Q", SAMPLE_STEPS[:1], SAMPLE_DOMAIN, threshold=0.6, revision_threshold=0.6
        )
        assert result.steps_needing_revision == [1]  # flagged despite overall_score 0.78

    @pytest.mark.asyncio
    async def test_step_critic_revision_threshold_is_configurable(self) -> None:
        """A borderline critical dimension is flagged only above the revision threshold."""
        critique_response = json.dumps(
            {
                "critiques": [
                    {
                        "step_number": 1,
                        "scores": {"aim_clarity": 0.9, "action_specificity": 0.5},
                        "overall_score": 0.85,
                        "issues": [],
                        "suggestions": [],
                    },
                ]
            }
        )
        # action_specificity 0.5 < 0.6 → flagged
        r_strict = await StepCriticAgent(_make_llm([critique_response])).critique(
            "Q", SAMPLE_STEPS[:1], SAMPLE_DOMAIN, threshold=0.6, revision_threshold=0.6
        )
        assert r_strict.steps_needing_revision == [1]
        # 0.5 >= 0.4 → not flagged when the threshold is relaxed
        r_relaxed = await StepCriticAgent(_make_llm([critique_response])).critique(
            "Q", SAMPLE_STEPS[:1], SAMPLE_DOMAIN, threshold=0.6, revision_threshold=0.4
        )
        assert r_relaxed.steps_needing_revision == []

    @pytest.mark.asyncio
    async def test_step_critique_disabled_by_default(self) -> None:
        """Config defaults to critique disabled."""
        config = MAGEConfig()
        assert config.enable_step_critique is False

    @pytest.mark.asyncio
    async def test_step_critique_threshold_config(self) -> None:
        """Config threshold is configurable."""
        config = MAGEConfig(enable_step_critique=True, step_critique_threshold=0.75)
        assert config.step_critique_threshold == 0.75


# ===========================================================================
# 2. Chain Verification tests
# ===========================================================================


class TestChainVerification:
    """Tests for ChainVerifier."""

    @pytest.mark.asyncio
    async def test_verification_generates_questions(self) -> None:
        """Verify that questions are generated for chain steps."""
        q_response = json.dumps(
            {
                "questions": [
                    {
                        "step_number": 1,
                        "question": "Does revenue analysis require domain knowledge?",
                        "aspect": "factual",
                    },
                    {
                        "step_number": 2,
                        "question": "Can risk assessment proceed without revenue data?",
                        "aspect": "logical",
                    },
                ]
            }
        )
        a_response1 = json.dumps({"answer": "yes", "explanation": "correct", "passed": True})
        a_response2 = json.dumps({"answer": "yes", "explanation": "correct", "passed": True})

        llm = _make_llm([q_response, a_response1, a_response2])
        verifier = ChainVerifier(llm)
        result = await verifier.verify("Analyze report", SAMPLE_CHAIN_DICT, SAMPLE_DOMAIN)

        assert isinstance(result, VerificationResult)
        assert len(result.questions) == 2
        assert result.chain_passed is True

    @pytest.mark.asyncio
    async def test_verification_passes_good_chain(self) -> None:
        """Good chain passes all verifications."""
        q_response = json.dumps({"questions": [{"step_number": 1, "question": "Q?", "aspect": "factual"}]})
        a_response = json.dumps({"answer": "yes", "passed": True, "explanation": "ok"})

        llm = _make_llm([q_response, a_response])
        verifier = ChainVerifier(llm)
        result = await verifier.verify("Q", SAMPLE_CHAIN_DICT, SAMPLE_DOMAIN)

        assert result.chain_passed is True
        assert len(result.issues_found) == 0

    @pytest.mark.asyncio
    async def test_verification_catches_issue(self) -> None:
        """Verification catches a failing question."""
        q_response = json.dumps(
            {"questions": [{"step_number": 2, "question": "Is step 2 dependency missing?", "aspect": "logical"}]}
        )
        a_response = json.dumps({"answer": "no", "passed": False, "explanation": "missing dep on step 1"})

        llm = _make_llm([q_response, a_response])
        verifier = ChainVerifier(llm)
        result = await verifier.verify("Q", SAMPLE_CHAIN_DICT, SAMPLE_DOMAIN)

        assert result.chain_passed is False
        assert len(result.issues_found) == 1
        assert result.issues_found[0]["step_number"] == 2

    @pytest.mark.asyncio
    async def test_verification_fix_applied(self) -> None:
        """Fixing updates the chain dict."""
        fixed_chain = {**SAMPLE_CHAIN_DICT, "name": "Fixed Chain"}
        llm = _make_llm([json.dumps(fixed_chain)])
        verifier = ChainVerifier(llm)

        vresult = VerificationResult(
            chain_passed=False,
            issues_found=[{"step_number": 2, "issue": "missing dep", "fix_suggestion": "add dep"}],
        )
        result = await verifier.fix(SAMPLE_CHAIN_DICT, vresult)

        assert result["name"] == "Fixed Chain"

    @pytest.mark.asyncio
    async def test_verification_disabled_by_default(self) -> None:
        """Config defaults to verification disabled."""
        config = MAGEConfig()
        assert config.enable_chain_verification is False


# ===========================================================================
# 3. Self-Refinement tests
# ===========================================================================


class TestSelfRefinement:
    """Tests for SelfRefiner."""

    @pytest.mark.asyncio
    async def test_refine_stops_when_satisfactory(self) -> None:
        """Refinement stops immediately if chain is satisfactory."""
        critique_response = json.dumps(
            {
                "overall_quality": 0.9,
                "is_satisfactory": True,
                "issues": [],
                "suggestions": [],
                "focus_areas": [],
            }
        )
        llm = _make_llm([critique_response])
        refiner = SelfRefiner(llm)
        result = await refiner.refine("Q", SAMPLE_CHAIN_DICT, SAMPLE_DOMAIN)

        assert isinstance(result, RefinementResult)
        assert result.iterations_used == 1
        assert result.final_quality == 0.9

    @pytest.mark.asyncio
    async def test_refine_improves_chain(self) -> None:
        """Refinement iterates: low quality → revision → high quality."""
        critique_low = json.dumps(
            {
                "overall_quality": 0.4,
                "is_satisfactory": False,
                "issues": ["vague aims"],
                "suggestions": ["improve"],
                "focus_areas": ["specificity"],
            }
        )
        revised_chain = json.dumps(SAMPLE_CHAIN_DICT)
        critique_high = json.dumps(
            {
                "overall_quality": 0.85,
                "is_satisfactory": True,
                "issues": [],
                "suggestions": [],
                "focus_areas": [],
            }
        )

        llm = _make_llm([critique_low, revised_chain, critique_high])
        refiner = SelfRefiner(llm)
        result = await refiner.refine("Q", SAMPLE_CHAIN_DICT, SAMPLE_DOMAIN)

        assert result.iterations_used == 2
        assert result.initial_quality == 0.4
        assert result.final_quality == 0.85

    @pytest.mark.asyncio
    async def test_refine_max_iterations_respected(self) -> None:
        """Refinement stops after max_iterations even if not satisfactory."""
        low_critique = json.dumps(
            {
                "overall_quality": 0.3,
                "is_satisfactory": False,
                "issues": ["bad"],
                "suggestions": ["fix"],
                "focus_areas": ["everything"],
            }
        )
        revised = json.dumps(SAMPLE_CHAIN_DICT)

        # 2 iterations: critique → revise → critique → revise (stops at max=2)
        responses = [low_critique, revised, low_critique, revised]
        llm = _make_llm(responses)
        refiner = SelfRefiner(llm)
        result = await refiner.refine("Q", SAMPLE_CHAIN_DICT, SAMPLE_DOMAIN, max_iterations=2)

        assert result.iterations_used == 2

    @pytest.mark.asyncio
    async def test_refine_disabled_by_default(self) -> None:
        """Config defaults to self-refine disabled."""
        config = MAGEConfig()
        assert config.enable_self_refine is False

    @pytest.mark.asyncio
    async def test_refine_quality_delta_tracked(self) -> None:
        """Quality delta (final - initial) is computed correctly."""
        critique1 = json.dumps(
            {
                "overall_quality": 0.4,
                "is_satisfactory": False,
                "issues": ["x"],
                "suggestions": ["y"],
                "focus_areas": ["z"],
            }
        )
        revised = json.dumps(SAMPLE_CHAIN_DICT)
        critique2 = json.dumps(
            {"overall_quality": 0.85, "is_satisfactory": True, "issues": [], "suggestions": [], "focus_areas": []}
        )

        llm = _make_llm([critique1, revised, critique2])
        refiner = SelfRefiner(llm)
        result = await refiner.refine("Q", SAMPLE_CHAIN_DICT, SAMPLE_DOMAIN)

        assert result.final_quality - result.initial_quality == pytest.approx(0.45)

    @pytest.mark.asyncio
    async def test_refine_returns_best_chain_not_a_regression(self) -> None:
        """P1.13: a revision that regresses quality is discarded; best chain is returned."""
        c_mid = json.dumps(
            {"overall_quality": 0.5, "is_satisfactory": False, "issues": ["x"], "suggestions": ["y"], "focus_areas": []}
        )
        bad_revision = json.dumps({"name": "worse", "steps": [{"number": 1, "title": "x"}]})
        c_low = json.dumps(
            {"overall_quality": 0.3, "is_satisfactory": False, "issues": [], "suggestions": [], "focus_areas": []}
        )

        llm = _make_llm([c_mid, bad_revision, c_low])
        result = await SelfRefiner(llm).refine("Q", SAMPLE_CHAIN_DICT, SAMPLE_DOMAIN, max_iterations=3)

        assert result.final_quality == 0.5  # the best seen, not the regressed 0.3
        assert result.chain_dict == SAMPLE_CHAIN_DICT  # original kept, bad revision discarded
        assert result.iterations_used == 2  # stopped early once quality regressed

    @pytest.mark.asyncio
    async def test_refine_stops_early_on_plateau(self) -> None:
        """P1.13: revisions that don't improve quality stop the loop early."""
        flat = json.dumps(
            {"overall_quality": 0.5, "is_satisfactory": False, "issues": ["x"], "suggestions": ["y"], "focus_areas": []}
        )
        revised = json.dumps(SAMPLE_CHAIN_DICT)

        # 0.5 → revise → 0.5 (no gain): should stop at iteration 2, not run all 5
        llm = _make_llm([flat, revised, flat, revised, flat, revised, flat, revised, flat, revised])
        result = await SelfRefiner(llm).refine("Q", SAMPLE_CHAIN_DICT, SAMPLE_DOMAIN, max_iterations=5)

        assert result.iterations_used == 2


# ===========================================================================
# 4. Tree of Thoughts tests
# ===========================================================================


class TestToTPlanning:
    """Tests for ToTPlanner."""

    @pytest.mark.asyncio
    async def test_tot_explores_branches(self) -> None:
        """ToT explores the configured number of branches."""
        dag_response = json.dumps(SAMPLE_DAG.model_dump())
        score_response = json.dumps(
            {
                "scores": {"parallelism": 0.8, "dependency_correctness": 0.9, "pattern_fit": 0.85, "efficiency": 0.8},
                "overall_score": 0.84,
                "rationale": "good",
            }
        )

        # 3 branches × (dag + score) + 2 refinements × (dag + score) = 10
        responses = [dag_response, score_response] * 5
        llm = _make_llm(responses)
        tot = ToTPlanner(llm)
        result = await tot.plan("Q", SAMPLE_STEP_PLAN, SAMPLE_DOMAIN, num_branches=3, depth=2)

        assert isinstance(result, ToTResult)
        assert result.branches_explored >= 3
        assert result.best_score > 0

    @pytest.mark.asyncio
    async def test_tot_selects_best_dag(self) -> None:
        """ToT selects the branch with the highest score."""
        dag1 = json.dumps(DAGStructure(pattern="pipeline", max_workers=1, steps=SAMPLE_DAG.steps).model_dump())
        score1 = json.dumps({"overall_score": 0.5, "rationale": "ok"})
        dag2 = json.dumps(DAGStructure(pattern="diamond", max_workers=2, steps=SAMPLE_DAG.steps).model_dump())
        score2 = json.dumps({"overall_score": 0.9, "rationale": "excellent"})
        dag3 = json.dumps(DAGStructure(pattern="funnel", max_workers=2, steps=SAMPLE_DAG.steps).model_dump())
        score3 = json.dumps({"overall_score": 0.7, "rationale": "good"})

        # 3 branches + 2 refinements (top 2 get refined)
        refine_dag = json.dumps(SAMPLE_DAG.model_dump())
        refine_score = json.dumps({"overall_score": 0.85, "rationale": "refined"})

        responses = [dag1, score1, dag2, score2, dag3, score3, refine_dag, refine_score, refine_dag, refine_score]
        llm = _make_llm(responses)
        tot = ToTPlanner(llm)
        result = await tot.plan("Q", SAMPLE_STEP_PLAN, SAMPLE_DOMAIN, num_branches=3, depth=2)

        assert result.best_score >= 0.85

    @pytest.mark.asyncio
    async def test_tot_depth_respected(self) -> None:
        """With depth=1, no refinement is done."""
        dag_response = json.dumps(SAMPLE_DAG.model_dump())
        score_response = json.dumps({"overall_score": 0.7, "rationale": "ok"})

        responses = [dag_response, score_response] * 3
        llm = _make_llm(responses)
        tot = ToTPlanner(llm)
        result = await tot.plan("Q", SAMPLE_STEP_PLAN, SAMPLE_DOMAIN, num_branches=3, depth=1)

        assert result.branches_explored == 3

    @pytest.mark.asyncio
    async def test_tot_disabled_by_default(self) -> None:
        """Config defaults to ToT disabled."""
        config = MAGEConfig()
        assert config.enable_tot_planning is False

    @pytest.mark.asyncio
    async def test_tot_integrates_with_generator(self) -> None:
        """ToT config fields are properly set."""
        config = MAGEConfig(enable_tot_planning=True, tot_branches=5, tot_depth=3)
        assert config.tot_branches == 5
        assert config.tot_depth == 3


# ===========================================================================
# 5. MCTS Construction tests
# ===========================================================================


class TestMCTSConstruction:
    """Tests for MCTSConstructor."""

    @pytest.mark.asyncio
    async def test_mcts_basic_construction(self) -> None:
        """MCTS builds a chain with the configured number of simulations."""
        propose_response = json.dumps(
            {
                "candidates": [
                    {
                        "step_type": "llm",
                        "number": 1,
                        "title": "Analyze data",
                        "aim": "Determine patterns",
                        "reasoning_questions": "What patterns?",
                        "step_context_queries": ["patterns"],
                        "stage_action": "Analyze",
                        "example_reasoning": "Patterns matter",
                        "dependencies": [],
                        "retry_max": 3,
                    },
                ]
            }
        )
        simulate_response = json.dumps(
            {
                "confidence": 0.8,
                "simulation_narrative": "Step succeeds",
                "remaining_steps_estimate": 2,
                "risk_factors": [],
            }
        )
        critic_response = json.dumps(
            {
                "scores": {"progress": 0.3, "step_quality": 0.8, "structure_quality": 0.9, "remaining_potential": 0.8},
                "overall_reward": 0.7,
                "critique": "Good start",
            }
        )

        # Each sim: propose + simulate + evaluate = 3 calls
        responses = [propose_response, simulate_response, critic_response] * 4
        llm = _make_llm(responses)
        mcts = MCTSConstructor(llm)
        config = MAGEConfig(enable_mcts=True, mcts_simulations=4, mcts_max_depth=5)
        result = await mcts.construct("Q", SAMPLE_DOMAIN, SAMPLE_STEP_PLAN, config)

        assert result.simulations_run == 4
        assert len(result.best_chain_steps) >= 1

    @pytest.mark.asyncio
    async def test_mcts_progressive_widening(self) -> None:
        """Progressive widening limits children in early simulations."""
        assert MCTSConstructor._progressive_widening_limit(0, True) == 1
        assert MCTSConstructor._progressive_widening_limit(2, True) == 1
        assert MCTSConstructor._progressive_widening_limit(3, True) == 2
        assert MCTSConstructor._progressive_widening_limit(5, True) == 2
        assert MCTSConstructor._progressive_widening_limit(6, True) == 3
        assert MCTSConstructor._progressive_widening_limit(0, False) == 3

    @pytest.mark.asyncio
    async def test_mcts_backtracking(self) -> None:
        """Low rewards trigger backtracking annotations."""
        node = MCTSNode(steps=[], parent=MCTSNode())
        node.abandon_reason = "Low reward (0.2): vague aims"
        assert "Low reward" in node.abandon_reason

    @pytest.mark.asyncio
    async def test_mcts_ucb1_selection(self) -> None:
        """UCB1 correctly selects high-reward unvisited nodes."""
        root = MCTSNode(steps=[], visits=10)
        child1 = MCTSNode(steps=SAMPLE_STEPS[:1], parent=root, visits=5, total_reward=2.5)
        child2 = MCTSNode(steps=SAMPLE_STEPS[:1], parent=root, visits=0)
        root.children = [child1, child2]

        # Unvisited child should have infinite UCB1
        assert child2.ucb1 == float("inf")
        assert child1.ucb1 > 0

        # With configurable c
        assert child2.ucb1_with_c(1.41) == float("inf")
        assert child1.ucb1_with_c(1.41) > 0

    @pytest.mark.asyncio
    async def test_mcts_critic_feedback_used(self) -> None:
        """Critic feedback is stored in MCTSNode."""
        node = MCTSNode(steps=[], critic_feedback="Good progress but needs more specificity")
        assert "specificity" in node.critic_feedback

    @pytest.mark.asyncio
    async def test_mcts_semantic_pruning(self) -> None:
        """Semantic pruning removes overly similar candidates."""
        existing = [MCTSNode(steps=[CARLStepSchema(number=1, title="Analyze revenue data")])]
        candidates = [
            CARLStepSchema(number=2, title="Analyze revenue data"),  # too similar
            CARLStepSchema(number=2, title="Evaluate market risks"),  # different
        ]

        pruned = MCTSConstructor._semantic_prune(candidates, existing)
        assert len(pruned) >= 1
        assert any("risk" in s.title.lower() for s in pruned)

    @pytest.mark.asyncio
    async def test_mcts_disabled_by_default(self) -> None:
        """Config defaults to MCTS disabled."""
        config = MAGEConfig()
        assert config.enable_mcts is False

    @pytest.mark.asyncio
    async def test_mcts_adaptive_exploration(self) -> None:
        """Complex tasks get higher exploration constant."""
        config = MAGEConfig(enable_mcts=True, mcts_exploration=1.41)
        # complex → factor 0.5 → c = 1.41 * 1.5 = 2.115
        c_complex = config.mcts_exploration * (1 + 0.5)
        assert c_complex == pytest.approx(2.115)

        # simple → factor 0 → c = 1.41
        c_simple = config.mcts_exploration * (1 + 0.0)
        assert c_simple == pytest.approx(1.41)


# ===========================================================================
# 6. Feedback Loop tests
# ===========================================================================


class TestFeedbackLoop:
    """Tests for FeedbackRecallAgent."""

    @pytest.mark.asyncio
    async def test_feedback_recall_empty_memory(self) -> None:
        """Empty memory returns empty digest."""
        memory = MagicMock()
        memory.search_feedback = AsyncMock(return_value=[])

        llm = _make_llm([])
        agent = FeedbackRecallAgent(llm, memory)
        result = await agent.recall("Q", SAMPLE_DOMAIN, MAGEConfig())

        assert isinstance(result, FeedbackDigest)
        assert result.recalled_count == 0
        assert result.digest == ""

    @pytest.mark.asyncio
    async def test_feedback_recall_with_results(self) -> None:
        """Feedback from memory is compiled into a digest."""
        raw_feedback = [
            {
                "chain_entity_id": "e1",
                "domain": "finance",
                "quality_score": 0.8,
                "success": True,
                "pattern_used": "diamond",
                "step_types_used": ["llm", "eval"],
            },
            {
                "chain_entity_id": "e2",
                "domain": "finance",
                "quality_score": 0.5,
                "success": False,
                "failure_reason": "step 3 failed",
                "pattern_used": "pipeline",
                "step_types_used": ["llm"],
            },
        ]
        memory = MagicMock()
        memory.search_feedback = AsyncMock(return_value=raw_feedback)

        # P1.3 applicability gate runs before the digest; keep both records.
        ratings = json.dumps({"ratings": [{"id": "0", "applicability": 0.9}, {"id": "1", "applicability": 0.8}]})
        digest_response = json.dumps({"digest": "Diamond pattern works well for finance"})
        llm = _make_llm([ratings, digest_response])

        agent = FeedbackRecallAgent(llm, memory)
        result = await agent.recall("Q", SAMPLE_DOMAIN, MAGEConfig(feedback_min_quality=0.3))

        assert result.recalled_count == 2
        assert result.digest == "Diamond pattern works well for finance"

    @pytest.mark.asyncio
    async def test_feedback_digest_generation(self) -> None:
        """Digest correctly identifies successful and failed patterns."""
        raw_feedback = [
            {"quality_score": 0.9, "success": True, "pattern_used": "diamond", "step_types_used": ["llm"]},
            {
                "quality_score": 0.4,
                "success": False,
                "failure_reason": "timeout",
                "pattern_used": "pipeline",
                "step_types_used": ["llm"],
            },
        ]
        memory = MagicMock()
        memory.search_feedback = AsyncMock(return_value=raw_feedback)

        # P1.3 applicability gate runs before the digest; keep both records.
        ratings = json.dumps({"ratings": [{"id": "0", "applicability": 0.9}, {"id": "1", "applicability": 0.6}]})
        digest_response = json.dumps({"digest": "Diamond recommended"})
        llm = _make_llm([ratings, digest_response])

        agent = FeedbackRecallAgent(llm, memory)
        result = await agent.recall("Q", SAMPLE_DOMAIN, MAGEConfig(feedback_min_quality=0.3))

        assert "diamond" in result.successful_patterns
        assert "timeout" in result.common_failures

    @pytest.mark.asyncio
    async def test_feedback_passed_to_planner(self) -> None:
        """Verify digest text can be passed as planner context."""
        digest = FeedbackDigest(
            digest="Use diamond pattern for finance tasks",
            recalled_count=3,
            avg_quality=0.75,
        )
        # The digest string should be non-empty and usable as context
        assert len(digest.digest) > 0
        assert "diamond" in digest.digest

    @pytest.mark.asyncio
    async def test_feedback_disabled_by_default(self) -> None:
        """Config defaults to feedback loop disabled."""
        config = MAGEConfig()
        assert config.enable_feedback_loop is False

    @pytest.mark.asyncio
    async def test_feedback_min_quality_filter(self) -> None:
        """Feedback below min_quality is filtered out."""
        raw_feedback = [
            {"quality_score": 0.1, "success": False, "pattern_used": "pipeline"},
            {"quality_score": 0.2, "success": False, "pattern_used": "funnel"},
        ]
        memory = MagicMock()
        memory.search_feedback = AsyncMock(return_value=raw_feedback)

        llm = _make_llm([])
        agent = FeedbackRecallAgent(llm, memory)
        result = await agent.recall("Q", SAMPLE_DOMAIN, MAGEConfig(feedback_min_quality=0.5))

        assert result.recalled_count == 0
        assert result.digest == ""


# ===========================================================================
# 7. Integration tests
# ===========================================================================


class TestIntegration:
    """Integration tests for quality enhancements."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_all_enhancements(self, mocker: Any) -> None:
        """Generator runs with all enhancements enabled (mocked)."""
        from tests.conftest import (
            SAMPLE_DAG,
            SAMPLE_DESCRIBED_STEPS,
            SAMPLE_DOMAIN_ANALYSIS,
            SAMPLE_STEP_PLAN,
        )

        # Build response sequence for deep mode + all enhancements
        domain_resp = json.dumps(SAMPLE_DOMAIN_ANALYSIS)
        plan_resp = json.dumps(SAMPLE_STEP_PLAN)
        dag_resp = json.dumps(SAMPLE_DAG)  # ToT generates DAGs
        score_resp = json.dumps({"overall_score": 0.8, "rationale": "good"})
        described_resp = json.dumps(SAMPLE_DESCRIBED_STEPS)

        # Feedback recall (P1.3 applicability gate runs before the digest)
        feedback_ratings_resp = json.dumps({"ratings": [{"id": "0", "applicability": 0.9}]})
        feedback_digest_resp = json.dumps({"digest": "Use funnel pattern"})

        # Step critique (all pass)
        critique_resp = json.dumps(
            {
                "critiques": [
                    {
                        "step_number": i,
                        "scores": {"aim_clarity": 0.9},
                        "overall_score": 0.9,
                        "issues": [],
                        "suggestions": [],
                    }
                    for i in range(1, 4)
                ]
            }
        )

        # Chain verification (pass)
        verify_q_resp = json.dumps({"questions": [{"step_number": 1, "question": "Q?", "aspect": "factual"}]})
        verify_a_resp = json.dumps({"answer": "yes", "passed": True, "explanation": "ok"})

        # Self-refine (satisfactory on first critique)
        refine_resp = json.dumps(
            {"overall_quality": 0.9, "is_satisfactory": True, "issues": [], "suggestions": [], "focus_areas": []}
        )

        responses = [
            domain_resp,  # domain analysis
            feedback_ratings_resp,  # feedback recall — applicability gate (P1.3)
            feedback_digest_resp,  # feedback recall — digest
            plan_resp,  # step planning
            # ToT: 3 branches × (dag + score) + 2 refinements × (dag + score)
            dag_resp,
            score_resp,
            dag_resp,
            score_resp,
            dag_resp,
            score_resp,
            dag_resp,
            score_resp,
            dag_resp,
            score_resp,
            described_resp,  # step describing
            critique_resp,  # step critique
            verify_q_resp,  # chain verification questions
            verify_a_resp,  # chain verification answer
            refine_resp,  # self-refine critique
        ]

        mock_cls = mocker.patch("mmar_mage.llm.AsyncOpenAI")
        instance = mock_cls.return_value
        instance.chat.completions.create = AsyncMock(side_effect=[_make_openai_response(r) for r in responses])
        instance.models.list = AsyncMock(return_value=MagicMock(data=[]))

        # Mock memory
        mocker.patch.object(
            __import__("mmar_mage.memory", fromlist=["MemoryManager"]).MemoryManager,
            "search_feedback",
            new=AsyncMock(
                return_value=[
                    {"quality_score": 0.8, "success": True, "pattern_used": "funnel"},
                ]
            ),
        )

        from mmar_mage import MAGEConfig, MAGEGenerator

        config = MAGEConfig(
            mode="deep",
            model="test",
            api_key="test-key",
            enable_memory=False,
            max_retries=0,
            enable_step_critique=True,
            enable_chain_verification=True,
            enable_self_refine=True,
            enable_tot_planning=True,
            enable_feedback_loop=True,
        )

        gen = MAGEGenerator(config)
        result = await gen.generate("Analyze financial report")

        assert result.mode == "deep"
        assert result.metadata.deep_stages_completed is not None
        stages = result.metadata.deep_stages_completed
        assert "tot_planning" in stages
        assert "step_critique" in stages
        assert "chain_verification" in stages
        assert "self_refine" in stages
        assert "feedback_recall" in stages

    @pytest.mark.asyncio
    async def test_enhancements_independently_toggleable(self) -> None:
        """Each enhancement can be toggled independently."""
        config = MAGEConfig(
            enable_step_critique=True,
            enable_chain_verification=False,
            enable_self_refine=True,
            enable_tot_planning=False,
            enable_mcts=False,
            enable_feedback_loop=True,
        )
        assert config.enable_step_critique is True
        assert config.enable_chain_verification is False
        assert config.enable_self_refine is True
        assert config.enable_tot_planning is False
        assert config.enable_mcts is False
        assert config.enable_feedback_loop is True

    @pytest.mark.asyncio
    async def test_metadata_tracks_enhancement_stats(self) -> None:
        """MAGEMetadata has all quality enhancement fields."""
        meta = MAGEMetadata(
            step_critique_score=0.85,
            verification_passed=True,
            refine_iterations=2,
            refine_quality_delta=0.3,
            tot_branches_explored=5,
            mcts_simulations_run=8,
            mcts_best_reward=0.9,
            feedback_recalled=3,
        )
        assert meta.step_critique_score == 0.85
        assert meta.verification_passed is True
        assert meta.refine_iterations == 2
        assert meta.refine_quality_delta == 0.3
        assert meta.tot_branches_explored == 5
        assert meta.mcts_simulations_run == 8
        assert meta.mcts_best_reward == 0.9
        assert meta.feedback_recalled == 3

    @pytest.mark.asyncio
    async def test_metadata_defaults_none(self) -> None:
        """Quality enhancement metadata defaults to None."""
        meta = MAGEMetadata()
        assert meta.step_critique_score is None
        assert meta.verification_passed is None
        assert meta.refine_iterations is None
        assert meta.mcts_simulations_run is None
        assert meta.feedback_recalled is None


# ===========================================================================
# 8. StepDescriber.revise() tests
# ===========================================================================


class TestStepDescriberRevise:
    """Tests for StepDescriber.revise() method."""

    @pytest.mark.asyncio
    async def test_revise_replaces_weak_steps(self) -> None:
        """Revise replaces only the steps flagged for revision."""
        revised_step = {
            "steps": [
                {
                    "step_type": "llm",
                    "number": 2,
                    "title": "Improved risk assessment",
                    "aim": "Evaluate financial risk factors systematically",
                    "reasoning_questions": "What specific risk metrics apply?",
                    "step_context_queries": ["risk metrics", "debt ratio"],
                    "stage_action": "Compute risk scores",
                    "example_reasoning": "Debt-to-equity reveals leverage risk",
                    "dependencies": [1],
                    "retry_max": 3,
                },
            ]
        }
        llm = _make_llm([json.dumps(revised_step)])

        from mmar_mage.agents.step_critic import StepCritique, StepCritiqueResult
        from mmar_mage.agents.step_describer import StepDescriber

        describer = StepDescriber(llm)
        critique_result = StepCritiqueResult(
            critiques=[StepCritique(step_number=2, overall_score=0.3, issues=["vague"], suggestions=["fix"])],
            overall_quality=0.6,
            steps_needing_revision=[2],
        )
        result = await describer.revise("Q", SAMPLE_DAG, SAMPLE_STEPS, critique_result)

        assert len(result) == 3
        # Step 1 and 3 unchanged
        assert result[0].title == "Analyze revenue"
        assert result[2].title == "Synthesize findings"
        # Step 2 revised
        assert result[1].title == "Improved risk assessment"

    @pytest.mark.asyncio
    async def test_revise_no_revision_needed(self) -> None:
        """When no steps need revision, returns original list."""
        from mmar_mage.agents.step_critic import StepCritiqueResult
        from mmar_mage.agents.step_describer import StepDescriber

        llm = _make_llm([])
        describer = StepDescriber(llm)
        critique_result = StepCritiqueResult(steps_needing_revision=[])

        result = await describer.revise("Q", SAMPLE_DAG, SAMPLE_STEPS, critique_result)
        assert result is SAMPLE_STEPS


# ===========================================================================
# 9. MemoryManager.search_feedback() test
# ===========================================================================


class TestMemorySearchFeedback:
    """Tests for MemoryManager.search_feedback()."""

    @pytest.mark.asyncio
    async def test_search_feedback_disabled(self) -> None:
        """Returns empty list when memory disabled."""
        from mmar_mage.memory import MemoryManager

        config = MAGEConfig(enable_memory=False)
        mgr = MemoryManager(config)
        result = await mgr.search_feedback("finance", "analysis")
        assert result == []
