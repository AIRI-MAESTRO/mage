"""
MMAR MAGE (Mesh Architecture Generation Engine)

A library for generating initial CARL reasoning chains from text queries.
Supports two modes:

- **fast**: single LLM call, returns CARL JSON directly.
- **deep** (default): agentic pipeline with DomainAnalyzer, StepPlanner,
  DAGBuilder, and StepDescriber — each individually toggleable.
  Optionally preceded by MemoryResearchAgent and WebResearchAgent.

Quick start::

    from mmar_mage import generate_chain, MAGEConfig

    result = await generate_chain("Analyze customer feedback")
    print(result.chain_json)

Module structure:

- ``mmar_mage.generator``: :class:`MAGEGenerator` orchestrator
- ``mmar_mage.schemas``: Pydantic models for config, results, intermediates
- ``mmar_mage.agents``: Deep-mode agents (DomainAnalyzer, StepPlanner, etc.)
- ``mmar_mage.memory``: gigaevo-memory integration
- ``mmar_mage.llm``: OpenAI-compatible LLM client
- ``mmar_mage.cli``: Typer CLI
- ``mmar_mage.prompts``: LLM prompt templates
- ``mmar_mage.exceptions``: Typed exception hierarchy
"""

from __future__ import annotations

from .agents.capability_lookup_agent import CapabilityContext
from .agents.chain_editor import ChainEditPlanner
from .agents.chain_verifier import ChainVerifier, VerificationQuestion, VerificationResult
from .agents.feedback_recall_agent import FeedbackDigest, FeedbackRecallAgent, RecalledFeedback
from .agents.mcts_constructor import MCTSConstructor, MCTSNode, MCTSResult
from .agents.self_refiner import RefinementCritique, RefinementResult, SelfRefiner
from .agents.step_critic import StepCriticAgent, StepCritique, StepCritiqueResult
from .agents.template_assembler_agent import TemplateSkeleton
from .agents.tot_planner import ToTBranch, ToTPlanner, ToTResult
from .carl_export import to_carl_compat
from .chain_edit import apply_edit_plan
from .code_generator import CodeGenerator
from .cost import CostEstimate, StepCostEstimate, estimate_chain_cost, estimate_pipeline_cost, fits_in_budget
from .evolve import (
    BenchmarkSynthesisResult,
    ChainEvolver,
    ChainScore,
    EvolveConfig,
    EvolveGeneration,
    EvolveResult,
    Individual,
    SynthesizedTask,
)
from .exceptions import (
    MAGECancelledError,
    MAGEError,
    MAGELLMError,
    MAGEMemoryError,
    MAGEValidationError,
)
from .generator import MAGEGenerator, MAGEProgressCallback, validate_carl_json
from .memory import MemoryManager
from .schemas import (
    CARL_VALID_STEP_TYPES,
    VALID_STEP_TYPES,
    CARLChainSchema,
    CARLStepSchema,
    ChainEdit,
    ChainEditCandidate,
    ChainEditPlan,
    DAGStep,
    DAGStructure,
    DomainAnalysis,
    MAGEConfig,
    MAGEEditResult,
    MAGEMetadata,
    MAGEResult,
    MemoryHit,
    MemoryResearchResult,
    PlannedStep,
    StepPlan,
    WebResearchResult,
)
from .skills import (
    KnownSkill,
    SkillDiscoveryAgent,
    SkillRecommendation,
    SkillRegistry,
    SkillSearchResult,
    default_registry,
)

__version__ = "0.1.3"

__all__ = [
    # Core
    "MAGEGenerator",
    "MAGEProgressCallback",
    "validate_carl_json",
    # Schemas
    "MAGEConfig",
    "MAGEResult",
    "MAGEMetadata",
    "DomainAnalysis",
    "StepPlan",
    "PlannedStep",
    "DAGStructure",
    "DAGStep",
    "CARLChainSchema",
    "CARLStepSchema",
    "MemoryHit",
    "MemoryResearchResult",
    "VALID_STEP_TYPES",
    "CARL_VALID_STEP_TYPES",
    "WebResearchResult",
    "to_carl_compat",
    # Edit chain mode (NL-driven targeted editing)
    "MAGEEditResult",
    "ChainEdit",
    "ChainEditPlan",
    "ChainEditCandidate",
    "ChainEditPlanner",
    "apply_edit_plan",
    "edit_chain",
    # Cost estimation
    "CostEstimate",
    "StepCostEstimate",
    "estimate_chain_cost",
    "estimate_pipeline_cost",
    "fits_in_budget",
    # Skill discovery
    "KnownSkill",
    "SkillDiscoveryAgent",
    "SkillRecommendation",
    "SkillRegistry",
    "SkillSearchResult",
    "default_registry",
    # Evolve mode
    "ChainEvolver",
    "EvolveConfig",
    "EvolveResult",
    "EvolveGeneration",
    "Individual",
    "ChainScore",
    "BenchmarkSynthesisResult",
    "SynthesizedTask",
    # Quality enhancement agents / models
    "StepCriticAgent",
    "StepCritique",
    "StepCritiqueResult",
    "ChainVerifier",
    "VerificationQuestion",
    "VerificationResult",
    "SelfRefiner",
    "RefinementCritique",
    "RefinementResult",
    "ToTPlanner",
    "ToTBranch",
    "ToTResult",
    "MCTSConstructor",
    "MCTSNode",
    "MCTSResult",
    "FeedbackRecallAgent",
    "FeedbackDigest",
    "RecalledFeedback",
    # Agents / models
    "CapabilityContext",
    "TemplateSkeleton",
    "CodeGenerator",
    # Memory
    "MemoryManager",
    # Exceptions
    "MAGEError",
    "MAGEValidationError",
    "MAGELLMError",
    "MAGEMemoryError",
    "MAGECancelledError",
    # Convenience
    "generate_chain",
]


async def generate_chain(query: str, config: MAGEConfig | None = None) -> MAGEResult:
    """Convenience function to generate a CARL chain.

    Parameters
    ----------
    query : str
        Text query describing the desired reasoning task.
    config : MAGEConfig | None
        Generation config. Uses defaults (deep mode, gpt-4o) if ``None``.

    Returns
    -------
    MAGEResult
        Generated chain with metadata.

    Examples
    --------
    >>> result = await generate_chain("Classify customer reviews")
    >>> print(result.metadata.num_steps)
    5
    """
    generator = MAGEGenerator(config=config or MAGEConfig())
    return await generator.generate(query)


async def edit_chain(
    instruction: str,
    *,
    entity_id: str | None = None,
    chain: dict | None = None,
    channel: str = "latest",
    save: bool = False,
    config: MAGEConfig | None = None,
) -> MAGEEditResult:
    """Convenience wrapper around :meth:`MAGEGenerator.edit`.

    Parameters
    ----------
    instruction : str
        What to change, in natural language.
    entity_id : str | None
        Id of the saved chain to edit. When omitted (and no ``chain`` is given),
        MAGE resolves the chain from ``instruction`` via memory search.
    chain : dict | None
        An in-memory chain to edit directly (skips resolution).
    channel : str
        Version channel to load from / save to.
    save : bool
        Persist the edited chain as a new version when an id is known.
    config : MAGEConfig | None
        Generation config. Uses defaults if ``None``.

    Returns
    -------
    MAGEEditResult

    Examples
    --------
    >>> result = await edit_chain("add a validation step", entity_id="abc123")
    >>> print(result.summary)
    """
    generator = MAGEGenerator(config=config or MAGEConfig())
    return await generator.edit(
        instruction, chain=chain, entity_id=entity_id, channel=channel, save=save
    )
