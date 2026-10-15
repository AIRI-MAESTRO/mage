"""
Deep-mode agents for the MAGE pipeline.

Each agent handles one stage of the agentic chain-generation workflow:
MemoryResearchAgent → WebResearchAgent → DomainAnalyzer →
CapabilityLookupAgent → TemplateAssemblerAgent → FeedbackRecallAgent →
StepPlanner → DAGBuilder/ToTPlanner → StepDescriber/MCTSConstructor →
StepCriticAgent → ChainVerifier → SelfRefiner.
"""

from __future__ import annotations

from .capability_lookup_agent import CapabilityContext, CapabilityLookupAgent
from .chain_verifier import ChainVerifier, VerificationQuestion, VerificationResult
from .dag_builder import DAGBuilder
from .domain_analyzer import DomainAnalyzer
from .feedback_recall_agent import FeedbackDigest, FeedbackRecallAgent, RecalledFeedback
from .mcts_constructor import MCTSConstructor, MCTSNode, MCTSResult
from .memory_research_agent import MemoryResearchAgent
from .self_refiner import RefinementCritique, RefinementResult, SelfRefiner
from .step_critic import StepCriticAgent, StepCritique, StepCritiqueResult
from .step_describer import StepDescriber
from .step_planner import StepPlanner
from .template_assembler_agent import TemplateAssemblerAgent, TemplateSkeleton
from .tot_planner import ToTBranch, ToTPlanner, ToTResult
from .web_research_agent import WebResearchAgent

__all__ = [
    "CapabilityContext",
    "CapabilityLookupAgent",
    "ChainVerifier",
    "DAGBuilder",
    "DomainAnalyzer",
    "FeedbackDigest",
    "FeedbackRecallAgent",
    "MCTSConstructor",
    "MCTSNode",
    "MCTSResult",
    "MemoryResearchAgent",
    "RecalledFeedback",
    "RefinementCritique",
    "RefinementResult",
    "SelfRefiner",
    "StepCritique",
    "StepCriticAgent",
    "StepCritiqueResult",
    "StepDescriber",
    "StepPlanner",
    "TemplateAssemblerAgent",
    "TemplateSkeleton",
    "ToTBranch",
    "ToTPlanner",
    "ToTResult",
    "VerificationQuestion",
    "VerificationResult",
    "WebResearchAgent",
]
