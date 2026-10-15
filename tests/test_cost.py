"""Tests for cost estimation (mmar_mage/cost.py)."""

from __future__ import annotations

import pytest

from mmar_mage import estimate_chain_cost, fits_in_budget

PRICING = {
    "gpt-4o-mini": (0.15 / 1000, 0.60 / 1000),  # placeholder per-1k-tokens prices
    "gpt-4o": (5.0 / 1000, 15.0 / 1000),
}


class TestEstimateBasic:
    def test_pure_llm_chain(self) -> None:
        chain = {
            "steps": [
                {"number": 1, "title": "A", "step_type": "llm", "aim": "do A", "dependencies": []},
                {"number": 2, "title": "B", "step_type": "llm", "aim": "do B", "dependencies": [1]},
            ]
        }
        est = estimate_chain_cost(chain, pricing=PRICING, default_model="gpt-4o-mini")
        assert est.total_input_tokens > 0
        assert est.total_output_tokens > 0
        assert est.total_cost_usd > 0
        assert len(est.steps) == 2
        assert all(row.calls_llm for row in est.steps)

    def test_non_llm_steps_zero_cost(self) -> None:
        chain = {
            "steps": [
                {
                    "number": 1,
                    "title": "Save",
                    "step_type": "memory",
                    "memory_operation": "write",
                    "memory_key": "k",
                    "dependencies": [],
                },
                {"number": 2, "title": "Tool", "step_type": "tool", "tool_name": "t", "dependencies": [1]},
                {"number": 3, "title": "Wait", "step_type": "human_input", "human_prompt": "?", "dependencies": [2]},
            ]
        }
        est = estimate_chain_cost(chain, pricing=PRICING, default_model="gpt-4o-mini")
        assert all(not row.calls_llm for row in est.steps)
        assert est.total_cost_usd == 0.0

    def test_no_pricing_marks_each_row(self) -> None:
        chain = {"steps": [{"number": 1, "title": "A", "step_type": "llm", "aim": "do A", "dependencies": []}]}
        est = estimate_chain_cost(chain, pricing=None, default_model="gpt-4o-mini")
        assert est.total_cost_usd == 0.0
        # No pricing dict at all → pricing_missing is False (caller didn't ask).
        # pricing_complete should be True only when pricing is provided.
        assert est.pricing_complete is False


class TestAgentFeaturesCosts:
    def test_evaluation_llm_method_counts(self) -> None:
        chain = {
            "steps": [
                {"number": 1, "title": "Out", "step_type": "llm", "aim": "produce", "dependencies": []},
                {
                    "number": 2,
                    "title": "Judge",
                    "step_type": "evaluation",
                    "evaluation_method": "llm",
                    "evaluation_target_step": 1,
                    "dependencies": [1],
                },
                {
                    "number": 3,
                    "title": "Rule",
                    "step_type": "evaluation",
                    "evaluation_method": "rule",
                    "evaluation_target_step": 1,
                    "dependencies": [1],
                },
            ]
        }
        est = estimate_chain_cost(chain, pricing=PRICING, default_model="gpt-4o-mini")
        # Rule-based evaluation doesn't call LLM
        rule_row = next(r for r in est.steps if r.step_number == 3)
        assert rule_row.calls_llm is False
        # LLM-judge evaluation does
        judge_row = next(r for r in est.steps if r.step_number == 2)
        assert judge_row.calls_llm is True
        assert judge_row.estimated_calls == 1

    def test_parallel_sampling_n_calls(self) -> None:
        chain = {
            "steps": [
                {
                    "number": 1,
                    "title": "Vote",
                    "step_type": "parallel_sampling",
                    "sampling_n": 7,
                    "sampling_aggregation": "vote",
                    "dependencies": [],
                }
            ]
        }
        est = estimate_chain_cost(chain, pricing=PRICING, default_model="gpt-4o-mini")
        assert est.steps[0].estimated_calls == 7

    def test_parallel_sampling_best_of_adds_judge(self) -> None:
        chain = {
            "steps": [
                {
                    "number": 1,
                    "title": "BestOf",
                    "step_type": "parallel_sampling",
                    "sampling_n": 4,
                    "sampling_aggregation": "best_of",
                    "dependencies": [],
                }
            ]
        }
        est = estimate_chain_cost(chain, pricing=PRICING, default_model="gpt-4o-mini")
        # n samples + 1 judge call
        assert est.steps[0].estimated_calls == 5

    def test_agent_skill_modes(self) -> None:
        chain = {
            "steps": [
                {
                    "number": 1,
                    "title": "SkillScript",
                    "step_type": "agent_skill",
                    "skill": "local:///abs/skill",
                    "skill_task": "t",
                    "skill_execution_mode": "script",
                    "dependencies": [],
                },
                {
                    "number": 2,
                    "title": "SkillLLM",
                    "step_type": "agent_skill",
                    "skill": "local:///abs/skill",
                    "skill_task": "t",
                    "skill_execution_mode": "llm",
                    "dependencies": [1],
                },
                {
                    "number": 3,
                    "title": "SkillAgent",
                    "step_type": "agent_skill",
                    "skill": "local:///abs/skill",
                    "skill_task": "t",
                    "skill_execution_mode": "llm_agent",
                    "skill_llm_max_iterations": 12,
                    "dependencies": [2],
                },
            ]
        }
        est = estimate_chain_cost(chain, pricing=PRICING, default_model="gpt-4o-mini")
        assert est.steps[0].calls_llm is False  # script mode
        assert est.steps[1].estimated_calls == 1  # llm mode = 1 call
        assert est.steps[2].estimated_calls == 12  # llm_agent up to N iterations

    def test_debate_count(self) -> None:
        chain = {
            "steps": [
                {
                    "number": 1,
                    "title": "Debate",
                    "step_type": "debate",
                    "debate_agents": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
                    "debate_rounds": 4,
                    "dependencies": [],
                }
            ]
        }
        est = estimate_chain_cost(chain, pricing=PRICING, default_model="gpt-4o-mini")
        # 4 rounds × 3 agents + 1 judge = 13
        assert est.steps[0].estimated_calls == 13


class TestBudgetGuard:
    def test_fits_under_threshold(self) -> None:
        chain = {"steps": [{"number": 1, "title": "A", "step_type": "llm", "aim": "do A", "dependencies": []}]}
        fits, est = fits_in_budget(chain, max_cost_usd=100.0, pricing=PRICING, default_model="gpt-4o-mini")
        assert fits is True
        assert est.total_cost_usd < 100.0

    def test_rejects_over_budget(self) -> None:
        chain = {
            "steps": [
                {
                    "number": 1,
                    "title": "Big",
                    "step_type": "parallel_sampling",
                    "sampling_n": 100,
                    "sampling_aggregation": "vote",
                    "dependencies": [],
                }
            ]
        }
        fits, est = fits_in_budget(chain, max_cost_usd=0.001, pricing=PRICING, default_model="gpt-4o")
        assert fits is False
        assert est.total_cost_usd > 0.001

    def test_token_limit(self) -> None:
        chain = {
            "steps": [
                {
                    "number": i,
                    "title": f"S{i}",
                    "step_type": "llm",
                    "aim": "a",
                    "dependencies": [i - 1] if i > 1 else [],
                }
                for i in range(1, 11)
            ]
        }
        fits, _ = fits_in_budget(chain, max_total_tokens=1000, default_model="gpt-4o-mini")
        assert fits is False  # 10 steps × ~512 output tokens > 1000

    def test_no_limits_always_fits(self) -> None:
        chain = {"steps": [{"number": 1, "title": "A", "step_type": "llm", "aim": "a", "dependencies": []}]}
        fits, _ = fits_in_budget(chain, default_model="gpt-4o-mini")
        assert fits is True


def test_format_table_returns_string() -> None:
    chain = {
        "steps": [
            {
                "number": 1,
                "title": "Title that is rather long indeed",
                "step_type": "llm",
                "aim": "do something",
                "dependencies": [],
            }
        ]
    }
    est = estimate_chain_cost(chain, pricing=PRICING, default_model="gpt-4o")
    out = est.format_table()
    assert "TOTAL" in out
    assert "llm" in out


@pytest.mark.parametrize("char_per_token", [-1, 0])
def test_invalid_char_per_token(char_per_token: int) -> None:
    with pytest.raises(ValueError):
        estimate_chain_cost({"steps": []}, char_per_token=char_per_token)
