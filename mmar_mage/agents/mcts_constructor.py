"""
MCTSConstructor — enhanced MCTS-based chain construction.

Builds CARL chains step-by-step using Monte Carlo Tree Search with
three sub-agents (Planner, Simulator, Critic).  Includes five
enhancements: adaptive exploration, progressive widening, critic
with explanation, semantic pruning, and backtracking annotations.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from ..llm import LLMClient
from ..prompts import MCTS_CRITIC_PROMPT, MCTS_PROPOSE_STEP_PROMPT, MCTS_SIMULATE_PROMPT
from ..schemas import CARLStepSchema, DomainAnalysis, MAGEConfig, StepPlan

logger = logging.getLogger(__name__)

STAGE = "mcts_construction"


# ---------------------------------------------------------------------------
# MCTS tree node (dataclass, not Pydantic — internal tree state)
# ---------------------------------------------------------------------------


@dataclass
class MCTSNode:
    """A node in the MCTS tree representing a partial chain.

    Parameters
    ----------
    steps : list[CARLStepSchema]
        Steps built so far along this path.
    parent : MCTSNode | None
        Parent node in the tree.
    children : list[MCTSNode]
        Child nodes (expanded candidates).
    visits : int
        Number of times this node was visited.
    total_reward : float
        Sum of rewards accumulated through this node.
    critic_feedback : str
        Natural language critique from the last evaluation.
    abandon_reason : str
        Reason this branch was abandoned (empty if not abandoned).
    """

    steps: list[CARLStepSchema] = field(default_factory=list)
    parent: MCTSNode | None = None
    children: list[MCTSNode] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    critic_feedback: str = ""
    abandon_reason: str = ""

    @property
    def ucb1(self) -> float:
        """Upper Confidence Bound score with adaptive exploration."""
        if self.visits == 0:
            return float("inf")
        if self.parent is None or self.parent.visits == 0:
            return self.total_reward / self.visits
        exploitation = self.total_reward / self.visits
        exploration = math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def ucb1_with_c(self, c: float) -> float:
        """UCB1 with configurable exploration constant.

        Parameters
        ----------
        c : float
            Exploration constant (higher = more exploration).

        Returns
        -------
        float
            UCB1 score.
        """
        if self.visits == 0:
            return float("inf")
        if self.parent is None or self.parent.visits == 0:
            return self.total_reward / self.visits
        exploitation = self.total_reward / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


class MCTSResult(BaseModel):
    """Result of MCTS chain construction.

    Parameters
    ----------
    best_chain_steps : list[CARLStepSchema]
        Steps of the best chain found.
    simulations_run : int
        Number of MCTS simulations executed.
    best_reward : float
        Reward of the best chain.
    tree_depth : int
        Maximum depth reached in the tree.
    backtrack_count : int
        Number of times a branch was abandoned.
    exploration_history : list[dict]
        Per-simulation history.
    """

    best_chain_steps: list[CARLStepSchema] = Field(default_factory=list)
    simulations_run: int = 0
    best_reward: float = 0.0
    tree_depth: int = 0
    backtrack_count: int = 0
    exploration_history: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Main constructor
# ---------------------------------------------------------------------------


class MCTSConstructor:
    """Build CARL chains via MCTS with three specialized sub-agents.

    Parameters
    ----------
    llm : LLMClient
        LLM client for all sub-agent calls.

    Examples
    --------
    >>> mcts = MCTSConstructor(llm_client)
    >>> result = await mcts.construct(query, domain_analysis, step_plan, config)
    >>> len(result.best_chain_steps)
    5
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def construct(
        self,
        query: str,
        domain_analysis: DomainAnalysis,
        step_plan: StepPlan,
        config: MAGEConfig,
    ) -> MCTSResult:
        """Build chain via MCTS with Planner/Simulator/Critic sub-agents.

        Parameters
        ----------
        query : str
            The user's original text query.
        domain_analysis : DomainAnalysis
            Domain analysis result.
        step_plan : StepPlan
            Step plan from the StepPlanner.
        config : MAGEConfig
            Configuration with MCTS parameters.

        Returns
        -------
        MCTSResult
            Best chain and exploration metadata.
        """
        num_sims = config.mcts_simulations
        max_depth = config.mcts_max_depth
        base_c = config.mcts_exploration
        progressive = config.mcts_progressive_widening

        # Adaptive exploration: adjust c based on complexity
        complexity_factor = {"complex": 0.5, "medium": 0.25, "simple": 0.0}.get(domain_analysis.complexity, 0.25)
        c = base_c * (1 + complexity_factor)

        logger.info(
            "MCTSConstructor: %d sims, max_depth=%d, c=%.3f (base=%.2f, complexity=%s)",
            num_sims,
            max_depth,
            c,
            base_c,
            domain_analysis.complexity,
        )

        root = MCTSNode(steps=[])
        exploration_history: list[dict] = []
        backtrack_count = 0

        for sim in range(num_sims):
            # 1. SELECT: traverse tree using UCB1
            node = self._select(root, c)

            # Check depth limit
            if len(node.steps) >= max_depth:
                continue

            # 2. EXPAND: Planner proposes candidates
            # Progressive widening: limit children in early simulations
            max_children = self._progressive_widening_limit(sim, progressive)

            candidates = await self._propose_steps(query, node.steps, step_plan, node.critic_feedback)

            # Semantic pruning: skip steps too similar to siblings
            candidates = self._semantic_prune(candidates, node.children)
            candidates = candidates[:max_children]

            best_child: MCTSNode | None = None
            best_reward = -1.0

            for candidate in candidates:
                new_steps = [*node.steps, candidate]

                # 3. SIMULATE
                await self._simulate(query, new_steps)

                # 4. EVALUATE
                eval_data = await self._evaluate(query, new_steps, domain_analysis)
                reward = eval_data.get("overall_reward", 0.0)
                critique = eval_data.get("critique", "")

                child = MCTSNode(
                    steps=new_steps,
                    parent=node,
                    critic_feedback=critique,
                )
                node.children.append(child)

                if reward > best_reward:
                    best_reward = reward
                    best_child = child

            # 5. BACKPROPAGATE
            if best_child is not None:
                self._backpropagate(best_child, best_reward)

                exploration_history.append(
                    {
                        "sim": sim + 1,
                        "action": best_child.steps[-1].title if best_child.steps else "",
                        "reward": best_reward,
                        "feedback": best_child.critic_feedback[:100],
                    }
                )

                # Backtracking annotation: if reward is very low, mark as abandoned
                if best_reward < 0.3 and node.parent is not None:
                    node.abandon_reason = f"Low reward ({best_reward:.2f}): {best_child.critic_feedback[:100]}"
                    backtrack_count += 1

        # Extract best path
        best_chain_steps = self._extract_best_path(root)
        tree_depth = self._max_depth(root)
        best_reward_final = max(
            (h["reward"] for h in exploration_history),
            default=0.0,
        )

        result = MCTSResult(
            best_chain_steps=best_chain_steps,
            simulations_run=num_sims,
            best_reward=best_reward_final,
            tree_depth=tree_depth,
            backtrack_count=backtrack_count,
            exploration_history=exploration_history,
        )

        logger.info(
            "MCTSConstructor: done, %d steps, best_reward=%.2f, backtracks=%d",
            len(best_chain_steps),
            best_reward_final,
            backtrack_count,
        )
        return result

    # ------------------------------------------------------------------
    # MCTS phases
    # ------------------------------------------------------------------

    @staticmethod
    def _select(root: MCTSNode, c: float) -> MCTSNode:
        """Select a leaf node using UCB1.

        Parameters
        ----------
        root : MCTSNode
            Root of the MCTS tree.
        c : float
            Exploration constant.

        Returns
        -------
        MCTSNode
            Selected leaf node.
        """
        node = root
        while node.children:
            # Pick child with highest UCB1
            node = max(node.children, key=lambda n: n.ucb1_with_c(c))
        return node

    async def _propose_steps(
        self,
        query: str,
        current_steps: list[CARLStepSchema],
        step_plan: StepPlan,
        critic_feedback: str,
    ) -> list[CARLStepSchema]:
        """Planner sub-agent: propose candidate next steps.

        Parameters
        ----------
        query : str
            User query.
        current_steps : list[CARLStepSchema]
            Steps built so far.
        step_plan : StepPlan
            Overall step plan.
        critic_feedback : str
            Feedback from previous critic evaluation.

        Returns
        -------
        list[CARLStepSchema]
            Candidate steps.
        """
        feedback_section = ""
        if critic_feedback:
            feedback_section = (
                f"Critic feedback from previous attempt:\n{critic_feedback}\nUse this to improve your proposals."
            )

        prompt = MCTS_PROPOSE_STEP_PROMPT.format(critic_feedback=feedback_section)

        steps_data = [s.model_dump(exclude_none=True) for s in current_steps]
        next_number = len(current_steps) + 1
        deps = [s.number for s in current_steps] if current_steps else []

        context = (
            f"User query: {query}\n\n"
            f"Step plan:\n{json.dumps(step_plan.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"Current partial chain ({len(current_steps)} steps):\n"
            f"{json.dumps(steps_data, ensure_ascii=False, indent=2)}\n\n"
            f"Next step number: {next_number}\n"
            f"Available dependency steps: {deps}"
        )

        data: dict[str, Any] = await self._llm.generate_json(
            system_prompt=prompt,
            user_prompt=context,
            stage=STAGE,
        )

        candidates: list[CARLStepSchema] = []
        for raw in data.get("candidates", []):
            raw["number"] = next_number
            candidates.append(CARLStepSchema.model_validate(raw))

        return candidates

    async def _simulate(
        self,
        query: str,
        steps: list[CARLStepSchema],
    ) -> dict[str, Any]:
        """Simulator sub-agent: predict execution outcome.

        Parameters
        ----------
        query : str
            User query.
        steps : list[CARLStepSchema]
            Partial chain including the new step.

        Returns
        -------
        dict[str, Any]
            Simulation result with confidence and narrative.
        """
        steps_data = [s.model_dump(exclude_none=True) for s in steps]
        context = (
            f"User query: {query}\n\n"
            f"Partial chain ({len(steps)} steps):\n"
            f"{json.dumps(steps_data, ensure_ascii=False, indent=2)}"
        )

        return await self._llm.generate_json(
            system_prompt=MCTS_SIMULATE_PROMPT,
            user_prompt=context,
            stage=STAGE,
        )

    async def _evaluate(
        self,
        query: str,
        steps: list[CARLStepSchema],
        domain_analysis: DomainAnalysis,
    ) -> dict[str, Any]:
        """Critic sub-agent: evaluate partial chain quality.

        Parameters
        ----------
        query : str
            User query.
        steps : list[CARLStepSchema]
            Partial chain to evaluate.
        domain_analysis : DomainAnalysis
            Domain analysis.

        Returns
        -------
        dict[str, Any]
            Evaluation with overall_reward and critique.
        """
        steps_data = [s.model_dump(exclude_none=True) for s in steps]
        context = (
            f"User query: {query}\n\n"
            f"Domain analysis:\n{json.dumps(domain_analysis.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"Partial chain ({len(steps)} steps):\n"
            f"{json.dumps(steps_data, ensure_ascii=False, indent=2)}"
        )

        return await self._llm.generate_json(
            system_prompt=MCTS_CRITIC_PROMPT,
            user_prompt=context,
            stage=STAGE,
        )

    @staticmethod
    def _backpropagate(node: MCTSNode, reward: float) -> None:
        """Backpropagate reward up the tree.

        Parameters
        ----------
        node : MCTSNode
            Node to start backpropagation from.
        reward : float
            Reward to propagate.
        """
        current: MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    # ------------------------------------------------------------------
    # Enhancement helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _progressive_widening_limit(sim: int, enabled: bool) -> int:
        """Calculate max children based on progressive widening.

        Parameters
        ----------
        sim : int
            Current simulation index (0-based).
        enabled : bool
            Whether progressive widening is enabled.

        Returns
        -------
        int
            Maximum number of children to expand.
        """
        if not enabled:
            return 3
        if sim < 3:
            return 1
        if sim < 6:
            return 2
        return 3

    @staticmethod
    def _semantic_prune(
        candidates: list[CARLStepSchema],
        existing_children: list[MCTSNode],
    ) -> list[CARLStepSchema]:
        """Remove candidates that are too similar to existing siblings.

        Uses simple substring overlap check (>90% title overlap).

        Parameters
        ----------
        candidates : list[CARLStepSchema]
            Proposed candidates.
        existing_children : list[MCTSNode]
            Already expanded siblings.

        Returns
        -------
        list[CARLStepSchema]
            Pruned candidates.
        """
        if not existing_children:
            return candidates

        existing_titles = set()
        for child in existing_children:
            if child.steps:
                existing_titles.add(child.steps[-1].title.lower())

        pruned: list[CARLStepSchema] = []
        for candidate in candidates:
            title_lower = candidate.title.lower()
            too_similar = False
            for existing in existing_titles:
                # Simple substring overlap check
                shorter = min(title_lower, existing, key=len)
                longer = max(title_lower, existing, key=len)
                if len(shorter) > 0 and shorter in longer:
                    overlap = len(shorter) / len(longer)
                    if overlap > 0.9:
                        too_similar = True
                        break
            if not too_similar:
                pruned.append(candidate)

        return pruned if pruned else candidates[:1]  # Always keep at least one

    # ------------------------------------------------------------------
    # Path extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_best_path(root: MCTSNode) -> list[CARLStepSchema]:
        """Extract the highest-reward path from root to leaf.

        Parameters
        ----------
        root : MCTSNode
            Root of the MCTS tree.

        Returns
        -------
        list[CARLStepSchema]
            Steps along the best path.
        """
        node = root
        while node.children:
            # Follow highest average reward
            node = max(
                node.children,
                key=lambda n: (n.total_reward / n.visits) if n.visits > 0 else 0.0,
            )
        return node.steps

    @staticmethod
    def _max_depth(root: MCTSNode) -> int:
        """Calculate maximum depth of the tree.

        Parameters
        ----------
        root : MCTSNode
            Root node.

        Returns
        -------
        int
            Maximum depth.
        """
        if not root.children:
            return 0
        return 1 + max(MCTSConstructor._max_depth(child) for child in root.children)
