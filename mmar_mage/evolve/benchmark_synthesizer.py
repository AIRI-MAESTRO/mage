"""BenchmarkSynthesizer — Stage 1 of evolve mode.

Given a user query, produce a small but diverse set of verifiable test cases.

Sources:
1. **LLM synthesis** — primary, always runs. Uses
   :data:`BENCHMARK_SYNTHESIZER_PROMPT` to generate concrete test instances
   covering different difficulty tiers and failure modes.
2. **Web** — optional, off by default. Uses :class:`WebResearchAgent` to find
   public benchmark items adjacent to the query (only if a web search API
   key is configured).
3. **Memory** — optional. Reuses tasks from past evolve runs in the same
   domain via :class:`MemoryManager`.

The result is :class:`BenchmarkSynthesisResult` with deduplicated tasks.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..agents.web_research_agent import WebResearchAgent
from ..llm import LLMClient
from ..memory import MemoryManager
from ..prompts import BENCHMARK_SYNTHESIZER_PROMPT
from ..schemas import DomainAnalysis, MAGEConfig
from .schemas import BenchmarkSynthesisResult, EvolveConfig, SynthesizedTask

logger = logging.getLogger(__name__)

STAGE = "benchmark_synthesis"


class BenchmarkSynthesizer:
    """Synthesize a per-query benchmark via LLM + (optional) web + memory."""

    def __init__(
        self,
        llm: LLMClient,
        memory: MemoryManager | None = None,
        web_research_agent: WebResearchAgent | None = None,
    ) -> None:
        self._llm = llm
        self._memory = memory
        self._web = web_research_agent

    async def synthesize(
        self,
        query: str,
        domain_analysis: DomainAnalysis,
        evolve_config: EvolveConfig,
        mage_config: MAGEConfig,
    ) -> BenchmarkSynthesisResult:
        """Run benchmark synthesis.

        Parameters
        ----------
        query : str
            User query — same one that will be evolved.
        domain_analysis : DomainAnalysis
            Domain context (used for memory lookup keys and LLM prompts).
        evolve_config : EvolveConfig
            Number of tasks, which sources to use, etc.
        mage_config : MAGEConfig
            For the web research path (API key, provider, ssl_verify).

        Returns
        -------
        BenchmarkSynthesisResult
            Deduplicated list of synthesized tasks plus rubric summary.
        """
        sources_used: list[str] = []

        # User-provided tasks always survive — the user knows their domain.
        tasks: list[SynthesizedTask] = list(evolve_config.user_provided_tasks)
        if tasks:
            sources_used.append("user")

        # 1. LLM synthesis — primary source.
        target_count = evolve_config.num_benchmark_tasks
        remaining = max(0, target_count - len(tasks))
        if remaining > 0:
            try:
                llm_tasks = await self._synthesize_via_llm(query, domain_analysis, remaining)
                tasks.extend(llm_tasks)
                sources_used.append("llm")
            except Exception as exc:
                logger.warning("BenchmarkSynthesizer: LLM synthesis failed: %s", exc)

        # 2. Memory recall — pulls past synthesized tasks for similar domain.
        if evolve_config.enable_memory_synthesis and self._memory is not None:
            try:
                mem_tasks = await self._recall_from_memory(domain_analysis)
                if mem_tasks:
                    tasks.extend(mem_tasks)
                    sources_used.append("memory")
            except Exception as exc:
                logger.warning("BenchmarkSynthesizer: memory recall failed: %s", exc)

        # 3. Web — only if explicitly enabled and provider configured.
        if evolve_config.enable_web_synthesis and self._web is not None and mage_config.web_search_api_key:
            try:
                web_tasks = await self._discover_via_web(query, domain_analysis, mage_config)
                if web_tasks:
                    tasks.extend(web_tasks)
                    sources_used.append("web")
            except Exception as exc:
                logger.warning("BenchmarkSynthesizer: web discovery failed: %s", exc)

        # Dedupe by input field; keep first occurrence (LLM > user > memory > web,
        # depending on which ran first — that's intentional, see ordering above).
        seen_inputs: set[str] = set()
        deduped: list[SynthesizedTask] = []
        for t in tasks:
            key = (t.input or "").strip().lower()
            if key and key not in seen_inputs:
                seen_inputs.add(key)
                deduped.append(t)

        # Re-number task_ids deterministically.
        for i, t in enumerate(deduped):
            t.task_id = f"synth_{i + 1:03d}"

        rubric_summary = self._collect_rubric_summary(deduped)
        return BenchmarkSynthesisResult(
            tasks=deduped,
            rubric_summary=rubric_summary,
            sources_used=sources_used,
        )

    # ------------------------------------------------------------------
    # LLM synthesis
    # ------------------------------------------------------------------

    async def _synthesize_via_llm(
        self,
        query: str,
        domain_analysis: DomainAnalysis,
        count: int,
    ) -> list[SynthesizedTask]:
        user_prompt = (
            f"Original query: {query}\n\n"
            f"Domain analysis:\n{json.dumps(domain_analysis.model_dump(), ensure_ascii=False, indent=2)}\n\n"
            f"Generate exactly {count} test cases."
        )
        data = await self._llm.generate_json(
            system_prompt=BENCHMARK_SYNTHESIZER_PROMPT,
            user_prompt=user_prompt,
            stage=STAGE,
        )

        raw_tasks = data.get("tasks", []) or []
        tasks: list[SynthesizedTask] = []
        for i, raw in enumerate(raw_tasks):
            if not isinstance(raw, dict):
                continue
            try:
                tasks.append(
                    SynthesizedTask(
                        task_id=raw.get("task_id") or f"synth_llm_{i + 1:03d}",
                        input=str(raw.get("input", "")).strip(),
                        expected_output=raw.get("expected_output"),
                        rubric=list(raw.get("rubric", []) or []),
                        difficulty=raw.get("difficulty", "medium"),
                        source="llm",
                        metadata=dict(raw.get("metadata", {}) or {}),
                    )
                )
            except Exception as exc:
                logger.debug("BenchmarkSynthesizer: skipping malformed task %d: %s", i, exc)
                continue

        # Drop tasks with empty input — they're useless for grading.
        return [t for t in tasks if t.input]

    # ------------------------------------------------------------------
    # Memory recall — best-effort, schema-tolerant.
    # ------------------------------------------------------------------

    async def _recall_from_memory(
        self,
        domain_analysis: DomainAnalysis,
    ) -> list[SynthesizedTask]:
        if self._memory is None:
            return []
        recalled: list[SynthesizedTask] = []

        try:
            hits = await self._memory.recall_benchmarks(
                domain=domain_analysis.domain,
                top_k=5,
            )
        except AttributeError:
            return []
        except Exception as exc:
            logger.debug("BenchmarkSynthesizer: memory.recall_benchmarks failed: %s", exc)
            return []

        for hit in hits or []:
            tasks_blob = hit.get("tasks") if isinstance(hit, dict) else None
            if not isinstance(tasks_blob, list):
                continue
            for raw in tasks_blob:
                if not isinstance(raw, dict):
                    continue
                try:
                    recalled.append(
                        SynthesizedTask(
                            task_id=raw.get("task_id") or "synth_mem",
                            input=str(raw.get("input", "")).strip(),
                            expected_output=raw.get("expected_output"),
                            rubric=list(raw.get("rubric", []) or []),
                            difficulty=raw.get("difficulty", "medium"),
                            source="memory",
                            metadata=dict(raw.get("metadata", {}) or {}),
                        )
                    )
                except Exception:
                    continue
        return [t for t in recalled if t.input]

    # ------------------------------------------------------------------
    # Web discovery — uses WebResearchAgent's digest to extract tasks.
    # ------------------------------------------------------------------

    async def _discover_via_web(
        self,
        query: str,
        domain_analysis: DomainAnalysis,
        mage_config: MAGEConfig,
    ) -> list[SynthesizedTask]:
        if self._web is None:
            return []
        result = await self._web.research(query, domain_analysis, mage_config)
        if not result.digest:
            return []

        # Ask the LLM to pull concrete test cases out of the digest.
        prompt = (
            f"Web digest about the query domain:\n\n{result.digest}\n\n"
            "Extract any concrete benchmark tasks present in the digest. "
            "For each task, produce input/expected_output/rubric in the same "
            "JSON shape as the BENCHMARK_SYNTHESIZER schema. If the digest "
            "contains no concrete tasks, return an empty tasks array."
        )
        try:
            data = await self._llm.generate_json(
                system_prompt=BENCHMARK_SYNTHESIZER_PROMPT,
                user_prompt=prompt,
                stage=STAGE,
            )
        except Exception as exc:
            logger.debug("BenchmarkSynthesizer: web extraction LLM call failed: %s", exc)
            return []

        web_tasks: list[SynthesizedTask] = []
        for i, raw in enumerate(data.get("tasks", []) or []):
            if not isinstance(raw, dict):
                continue
            input_text = str(raw.get("input", "")).strip()
            if not input_text:
                continue
            web_tasks.append(
                SynthesizedTask(
                    task_id=raw.get("task_id") or f"synth_web_{i + 1:03d}",
                    input=input_text,
                    expected_output=raw.get("expected_output"),
                    rubric=list(raw.get("rubric", []) or []),
                    difficulty=raw.get("difficulty", "medium"),
                    source="web",
                    metadata={**dict(raw.get("metadata", {}) or {}), "web_sources": result.sources},
                )
            )
        return web_tasks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_rubric_summary(tasks: list[SynthesizedTask]) -> str:
        """Summarize what the chain output as a whole must establish."""
        if not tasks:
            return ""
        items: list[str] = []
        for t in tasks:
            for r in t.rubric:
                if r and r not in items:
                    items.append(r)
        if not items:
            return ""
        head = items[:6]
        more = f" (+{len(items) - len(head)} more)" if len(items) > len(head) else ""
        return "; ".join(head) + more


def chain_dict_summary_for_judge(chain_dict: dict[str, Any]) -> str:
    """Compact one-line-per-step summary, used by ChainGrader prompts."""
    lines: list[str] = []
    for s in chain_dict.get("steps", []):
        lines.append(
            f"#{s.get('number')} [{s.get('step_type', 'llm')}] {s.get('title', '')} — aim: {s.get('aim', '')[:80]}"
        )
    return "\n".join(lines)
