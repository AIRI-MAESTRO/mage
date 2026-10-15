"""TopologySelectorAgent — pick a structural topology for a query (plan item P1.15).

Runs after the DomainAnalyzer and before the StepPlanner. Today MAGE mostly emits
linear pipelines because pattern choice is implicit; this agent makes it explicit
by ranking the built-in topology catalog against the query/domain and emitting
1–3 :class:`TopologyChoice`s. Graceful: any failure (or an empty/invalid result)
falls back to the default ``pipeline`` so generation never blocks on it.
"""

from __future__ import annotations

import logging

from ..llm import LLMClient
from ..prompts import TOPOLOGY_SELECTOR_PROMPT
from ..schemas import DomainAnalysis
from ..topology import (
    TopologyCandidates,
    TopologyChoice,
    default_choice,
    describe_for_prompt,
    get_topology,
)

logger = logging.getLogger(__name__)

STAGE = "topology_selection"


class TopologySelectorAgent:
    """Select 1–3 ranked topologies for a query.

    Parameters
    ----------
    llm : LLMClient
        Client used for the single ranking call.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def select(
        self,
        query: str,
        domain: DomainAnalysis,
        max_candidates: int = 3,
        topology_hint_demos: str = "",
    ) -> TopologyCandidates:
        """Return ranked topology candidates (best first).

        Never raises: on any failure returns a single-element ``pipeline``
        candidate so the caller can proceed unconstrained.
        """
        import json

        context = (
            f"User query: {query}\n\nDomain analysis:\n{json.dumps(domain.model_dump(), ensure_ascii=False, indent=2)}"
        )
        if topology_hint_demos:
            context += f"\n\n## Worked examples (for shape reference):\n{topology_hint_demos}"

        try:
            data = await self._llm.generate_json(
                system_prompt=TOPOLOGY_SELECTOR_PROMPT.format(topology_catalog=describe_for_prompt()),
                user_prompt=context,
                stage=STAGE,
            )
        except Exception as exc:
            logger.warning("TopologySelector: selection failed (non-fatal): %s", exc)
            return TopologyCandidates(candidates=[default_choice()])

        candidates = self._parse(data, max_candidates)
        if not candidates:
            logger.info("TopologySelector: no valid candidate — defaulting to pipeline")
            candidates = [default_choice()]
        logger.info("TopologySelector: chose %s", ", ".join(c.name for c in candidates))
        return TopologyCandidates(candidates=candidates)

    @staticmethod
    def _parse(data: dict, max_candidates: int) -> list[TopologyChoice]:
        """Validate raw candidates against the catalog; drop unknown names + dups."""
        raw = data.get("candidates", []) if isinstance(data, dict) else []
        out: list[TopologyChoice] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            topo = get_topology(item.get("name"))
            if topo is None or topo.name in seen:
                continue  # unknown name or duplicate
            seen.add(topo.name)
            try:
                conf = float(item.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            out.append(
                TopologyChoice(
                    name=topo.name,
                    rationale=str(item.get("rationale", "")).strip(),
                    confidence=max(0.0, min(1.0, conf)),
                )
            )
            if len(out) >= max_candidates:
                break
        return out
