"""Topology schemas (plan item P1.15).

A *topology* is a named structural pattern a reasoning chain can take (pipeline,
diamond, debate, …). These schemas describe the catalog entries and the
selector's output so downstream agents (StepPlanner, DAGBuilder) can reason about
the intended structure programmatically rather than inferring it implicitly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Topology(BaseModel):
    """One catalog entry — a named structural pattern with usage metadata.

    Parameters
    ----------
    name : str
        Stable identifier, e.g. ``"diamond"``.
    when_to_use : str
        One-line guidance on the query shape this pattern fits.
    required_step_types : list[str]
        Step types this topology characteristically emits.
    example_chain : str
        Short textual sketch of the step layout (for the selector prompt).
    complexity_score : float
        Rough authoring complexity, ``0`` (trivial) … ``1`` (heavy orchestration).
        Lets the selector trade structure against the simplicity bias (P1.10).
    """

    name: str
    when_to_use: str
    required_step_types: list[str] = Field(default_factory=list)
    example_chain: str = ""
    complexity_score: float = 0.5


class TopologyChoice(BaseModel):
    """A selector's ranked pick of a topology for a specific query."""

    name: str
    rationale: str = ""
    confidence: float = 0.5


class TopologyCandidates(BaseModel):
    """1–3 ranked topology choices for a query (best first)."""

    candidates: list[TopologyChoice] = Field(default_factory=list)

    @property
    def best(self) -> TopologyChoice | None:
        """The top-ranked choice, or ``None`` when empty."""
        return self.candidates[0] if self.candidates else None
