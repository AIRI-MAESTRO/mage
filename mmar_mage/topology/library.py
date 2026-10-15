"""Built-in topology catalog (plan item P1.15).

A registry of named structural patterns MAGE can target, each carrying enough
metadata (`when_to_use`, `required_step_types`, `complexity_score`, a sketch) for
the :class:`~mmar_mage.agents.topology_selector.TopologySelectorAgent` to choose
among them and for the StepPlanner / DAGBuilder to plan toward them.

Today MAGE mostly emits linear pipelines because the StepPlanner few-shots are
pipeline-heavy and pattern choice is implicit. This catalog makes the choice
explicit and programmatic.
"""

from __future__ import annotations

from .schemas import Topology, TopologyChoice

# Default fallback topology name — the safe, always-valid choice.
DEFAULT_TOPOLOGY = "pipeline"

BUILTIN_TOPOLOGIES: dict[str, Topology] = {
    "pipeline": Topology(
        name="pipeline",
        when_to_use="Sequential single-track reasoning; each step consumes the previous one.",
        required_step_types=["llm", "tool", "transform"],
        example_chain="1 → 2 → 3 (linear)",
        complexity_score=0.2,
    ),
    "diamond": Topology(
        name="diamond",
        when_to_use="One input fans out to N independent analyses that a synthesizer merges.",
        required_step_types=["llm", "aggregator"],
        example_chain="1 → (2a, 2b, 2c) → 3 (synthesize)",
        complexity_score=0.5,
    ),
    "funnel": Topology(
        name="funnel",
        when_to_use="N independent inputs/sources collapse into one synthesizer.",
        required_step_types=["llm", "aggregator"],
        example_chain="(1, 2, 3) → 4 (synthesize)",
        complexity_score=0.5,
    ),
    "map_reduce": Topology(
        name="map_reduce",
        when_to_use=(
            "The SAME analysis over many independent items, then reduced — "
            "N samples, several files, or one large file read/analysed in "
            "chunks (outline → per-chunk analyse fan-out → merge)."
        ),
        required_step_types=["parallel_sampling", "aggregator"],
        example_chain="outline(1) → map: read+analyse chunk (2 ×N) → reduce/merge(3)",
        complexity_score=0.6,
    ),
    "tree_of_thought": Topology(
        name="tree_of_thought",
        when_to_use="Exploratory reasoning with branching and backtracking.",
        required_step_types=["llm"],
        example_chain="1 → branch(2a, 2b) → evaluate → loop_back_to",
        complexity_score=0.7,
    ),
    "mixture_of_experts": Topology(
        name="mixture_of_experts",
        when_to_use="Heterogeneous sub-tasks needing routing to specialized handlers.",
        required_step_types=["supervisor", "llm"],
        example_chain="1 (route) → expert_a | expert_b → 3 (merge)",
        complexity_score=0.8,
    ),
    "debate": Topology(
        name="debate",
        when_to_use="Close-call or contested decisions where opposing arguments sharpen the answer.",
        required_step_types=["debate", "llm"],
        example_chain="1 → debate(pro, con) → 3 (judge)",
        complexity_score=0.6,
    ),
    "critique_revise_loop": Topology(
        name="critique_revise_loop",
        when_to_use="Output quality matters more than latency; iterate draft → critique → revise.",
        required_step_types=["llm", "evaluation"],
        example_chain="1 (draft) → 2 (critique) → revise → loop_back_to",
        complexity_score=0.5,
    ),
    "hierarchical": Topology(
        name="hierarchical",
        when_to_use="Multi-level orchestration: a supervisor of supervisors.",
        required_step_types=["supervisor", "llm"],
        example_chain="1 (lead) → (sup_a → ..., sup_b → ...) → 3",
        complexity_score=0.9,
    ),
    "orchestrator_with_rollback": Topology(
        name="orchestrator_with_rollback",
        when_to_use="High-stakes chains needing checkpoints and recovery on failure.",
        required_step_types=["llm", "tool"],
        example_chain="1 → 2 (checkpoint) → 3 → (on fail: rollback to 2)",
        complexity_score=0.8,
    ),
    "router": Topology(
        name="router",
        when_to_use="Classify the request and route it to exactly ONE specialized handler (no merge — lighter than mixture_of_experts).",
        required_step_types=["supervisor", "llm"],
        example_chain="1 (classify) → route → handler_x",
        complexity_score=0.4,
    ),
    "self_consistency": Topology(
        name="self_consistency",
        when_to_use="Reduce variance on a SINGLE hard question by sampling the same prompt N times independently and taking the consensus.",
        required_step_types=["parallel_sampling", "aggregator"],
        example_chain="1 → sample(2 ×N, same prompt) → vote(3)",
        complexity_score=0.5,
    ),
    "react": Topology(
        name="react",
        when_to_use="Interleave reasoning with tool actions: think, act (call a tool), observe the result, repeat until done.",
        required_step_types=["llm", "tool"],
        example_chain="1 (reason) → 2 (act/tool) → observe → loop_back_to",
        complexity_score=0.6,
    ),
    "chain_of_verification": Topology(
        name="chain_of_verification",
        when_to_use="Cut factual hallucination: draft an answer, generate verification questions, answer them, then revise to match.",
        required_step_types=["llm", "evaluation"],
        example_chain="1 (draft) → 2 (verify questions) → 3 (check) → 4 (revise)",
        complexity_score=0.6,
    ),
    "plan_and_execute": Topology(
        name="plan_and_execute",
        when_to_use="Multi-part requests: a planner decomposes the task into independent sub-tasks, each executed, then results assembled.",
        required_step_types=["llm", "agent_handoff", "aggregator"],
        example_chain="1 (plan) → (execute 2a, 2b, 2c) → 4 (assemble)",
        complexity_score=0.7,
    ),
    "human_in_the_loop": Topology(
        name="human_in_the_loop",
        when_to_use="A consequential or ambiguous action needs human approval or clarification before proceeding.",
        required_step_types=["human_input", "llm"],
        example_chain="1 → 2 (human approve/clarify?) → 3 (act)",
        complexity_score=0.5,
    ),
}


def get_topology(name: str | None) -> Topology | None:
    """Return the catalog entry for ``name`` (case-insensitive), or ``None``."""
    if not name:
        return None
    return BUILTIN_TOPOLOGIES.get(name.strip().lower())


def list_topologies() -> list[Topology]:
    """All catalog entries, simplest first (by complexity score)."""
    return sorted(BUILTIN_TOPOLOGIES.values(), key=lambda t: t.complexity_score)


def default_choice() -> TopologyChoice:
    """The deterministic fallback choice (pipeline) used when selection is off/fails."""
    return TopologyChoice(name=DEFAULT_TOPOLOGY, rationale="Default linear pipeline.", confidence=0.5)


def describe_for_prompt() -> str:
    """Render the catalog as a compact table for the selector prompt."""
    lines = ["name | when_to_use | step_types | complexity"]
    for t in list_topologies():
        types = ", ".join(t.required_step_types)
        lines.append(f"{t.name} | {t.when_to_use} | {types} | {t.complexity_score:.1f}")
    return "\n".join(lines)


def describe_patterns_for_dag() -> str:
    """Render the catalog as a ``- "name": when_to_use (shape)`` enum for the
    DAG-builder prompt — the single source of pattern names (P-4)."""
    return "\n".join(
        f'- "{t.name}": {t.when_to_use} (shape: {t.example_chain})' for t in list_topologies()
    )


def format_topology_hint(choice: TopologyChoice) -> str:
    """Render a chosen topology as a planning constraint for the StepPlanner/DAGBuilder.

    Returns an empty string when the choice is unknown or a plain pipeline (no
    constraint needed — that's the StepPlanner's default behavior anyway).
    """
    topo = get_topology(choice.name)
    if topo is None or topo.name == DEFAULT_TOPOLOGY:
        return ""
    types = ", ".join(topo.required_step_types)
    return (
        f"Target topology: {topo.name} — {topo.when_to_use}\n"
        f"Characteristic step types: {types}. Example layout: {topo.example_chain}.\n"
        "Structure the steps (and their dependencies) to realize this topology."
    )
