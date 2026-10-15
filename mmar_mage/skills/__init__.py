"""Skill discovery and recommendation for MAGE.

Lets MAGE pull AgentSkills from the internet and pick the right ones for
a given task. Used during deep-mode generation to inform StepPlanner when
an ``agent_skill`` step is a better fit than a hand-rolled ``llm`` step.

Architecture
------------

1. :class:`SkillRegistry` — curated catalog of known public skills with
   URIs and descriptions. Lives in :mod:`mmar_mage.skills.registry`.
2. :class:`SkillDiscoveryAgent` — LLM-driven pipeline:
   a. Decides whether the task plausibly benefits from any packaged skill.
   b. Searches the local registry by semantic match (LLM-rerank).
   c. If no good match, optionally falls back to web search for
      ``agent skills site:github.com`` style queries.
   d. Returns a digest of recommended skills with URIs + reasons.
3. The digest is passed into :class:`StepPlanner` as additional context so
   the generated chain can include ``agent_skill`` steps with concrete
   ``skill`` URIs.

Failure modes are non-fatal: if discovery times out, returns no
recommendations and the chain still gets generated without skills.
"""

from __future__ import annotations

from .discovery import (
    SkillDiscoveryAgent,
    SkillRecommendation,
    SkillSearchResult,
    format_surfaced_skills,
    surface_skills,
)
from .registry import KnownSkill, SkillRegistry, default_registry

__all__ = [
    "KnownSkill",
    "SkillDiscoveryAgent",
    "SkillRecommendation",
    "SkillRegistry",
    "SkillSearchResult",
    "default_registry",
    "format_surfaced_skills",
    "surface_skills",
]
