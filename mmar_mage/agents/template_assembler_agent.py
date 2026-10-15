"""
TemplateAssemblerAgent — searches gigaevo-memory for reusable chain templates.

Finds ChainTemplate entities matching (domain, task_type, complexity)
and selects the best by quality_score * success_rate.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from ..memory import MemoryManager
from ..schemas import MAGEConfig
from ..templates import select_skeleton

logger = logging.getLogger(__name__)


class TemplateSkeleton(BaseModel):
    """Suggested DAG skeleton from a matched template.

    Parameters
    ----------
    template_id : str | None
        Memory entity ID of the template.
    abstract_dag : list[dict[str, Any]]
        Abstract DAG structure from the template.
    step_type_sequence : list[str]
        Ordered step types from the template.
    quality_score : float
        Template quality score.
    success_rate : float
        Template success rate.
    """

    template_id: str | None = None
    abstract_dag: list[dict[str, Any]] = Field(default_factory=list)
    step_type_sequence: list[str] = Field(default_factory=list)
    quality_score: float = 0.0
    success_rate: float = 0.0


class TemplateAssemblerAgent:
    """Search memory for matching chain templates.

    Parameters
    ----------
    memory_manager : MemoryManager
        Memory manager for template search.
    config : MAGEConfig
        MAGE configuration.
    """

    def __init__(self, memory_manager: MemoryManager, config: MAGEConfig) -> None:
        self._memory = memory_manager
        self._config = config

    async def assemble(self, domain: str, task_type: str, complexity: str) -> TemplateSkeleton | None:
        """Search for and select the best matching template.

        Parameters
        ----------
        domain : str
            Knowledge domain.
        task_type : str
            Task type (analysis, generation, etc.).
        complexity : str
            Complexity level (simple, medium, complex).

        Returns
        -------
        TemplateSkeleton | None
            Best matching stored template, or — when none is found — a
            synthesized built-in skeleton matching the task type (P1.3). Returns
            ``None`` only on unexpected failure.
        """
        try:
            templates = await self._memory.search_templates(domain, task_type, complexity)
            if not templates:
                logger.info(
                    "TemplateAssemblerAgent: no stored template for %s/%s/%s — synthesizing a skeleton",
                    domain,
                    task_type,
                    complexity,
                )
                return self._synthesize_skeleton(domain, task_type)

            best = max(
                templates,
                key=lambda t: t.get("quality_score", 0.0) * t.get("success_rate", 0.0),
            )

            return TemplateSkeleton(
                template_id=best.get("entity_id"),
                abstract_dag=best.get("abstract_dag", []),
                step_type_sequence=best.get("step_type_sequence", []),
                quality_score=best.get("quality_score", 0.0),
                success_rate=best.get("success_rate", 0.0),
            )
        except Exception as exc:
            logger.warning("TemplateAssemblerAgent: assembly failed (non-fatal): %s", exc)
            return None

    @staticmethod
    def _synthesize_skeleton(domain: str, task_type: str) -> TemplateSkeleton:
        """Build a soft skeleton from the built-in library when memory misses (P1.3).

        ``quality_score``/``success_rate`` are left at 0.0 to mark this as
        unproven guidance (the StepPlanner treats the skeleton as a hint, not a
        template to copy), and ``template_id`` is namespaced ``builtin:<name>``.
        """
        skel = select_skeleton(task_type, domain)
        return TemplateSkeleton(
            template_id=f"builtin:{skel['name']}",
            abstract_dag=list(skel["abstract_dag"]),
            step_type_sequence=list(skel["step_type_sequence"]),
            quality_score=0.0,
            success_rate=0.0,
        )
