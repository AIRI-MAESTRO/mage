"""Topology library + selection (plan item P1.15).

Public surface for diverse-topology generation: the catalog of named structural
patterns, helpers to describe/select them, and the schemas the
``TopologySelectorAgent`` produces.
"""

from __future__ import annotations

from .library import (
    BUILTIN_TOPOLOGIES,
    DEFAULT_TOPOLOGY,
    default_choice,
    describe_for_prompt,
    format_topology_hint,
    get_topology,
    list_topologies,
)
from .schemas import Topology, TopologyCandidates, TopologyChoice

__all__ = [
    "BUILTIN_TOPOLOGIES",
    "DEFAULT_TOPOLOGY",
    "Topology",
    "TopologyCandidates",
    "TopologyChoice",
    "default_choice",
    "describe_for_prompt",
    "format_topology_hint",
    "get_topology",
    "list_topologies",
]
