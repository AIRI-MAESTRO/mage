"""ChainEvolver — the evolutionary search loop.

Orchestrates Stages 1-5 of evolve mode (see :mod:`mmar_mage.evolve`):

1. Synthesize benchmark from query.
2. Generate diverse initial population via :class:`MAGEGenerator`.
3. Grade every individual against the benchmark.
4. Apply mutation / crossover / LLM-repair operators; rank; iterate.
5. Stop on plateau / score threshold / max_generations.

The evolver is intentionally model-agnostic: any :class:`MAGEGenerator` works,
and you can swap the grader's LLM-judge for real :class:`ChainExecutor` runs
later without touching the engine.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import random
import uuid
from collections.abc import Callable
from typing import Any

from ..agents import DomainAnalyzer, WebResearchAgent
from ..carl_export import to_carl_compat
from ..cost import estimate_chain_cost, fits_in_budget
from ..generator import MAGEGenerator, validate_carl_json
from ..llm import LLMClient
from ..schemas import DomainAnalysis, MAGEConfig
from .benchmark_synthesizer import BenchmarkSynthesizer
from .grader import ChainGrader
from .operators import (
    change_step_type,
    crossover,
    insert_step,
    llm_repair,
    mutate_aim,
    reflective_mutate,
    remove_step,
    swap_steps,
)
from .schemas import (
    BenchmarkSynthesisResult,
    EvolveConfig,
    EvolveGeneration,
    EvolveResult,
    Individual,
)

logger = logging.getLogger(__name__)


# Operator registry: name → (kind, function)
# kind ∈ {"mutation", "crossover", "llm"} — controls call signature.
OperatorFn = Callable[..., Any]

# T25 / P-5: how far a chain the judge flagged as benchmark-gaming is demoted in
# ranking. Scores are ~0..1, so a 0.5 penalty reliably sinks a gamed chain below
# an honest one of comparable raw score, without hard-excluding it.
_GAMING_PENALTY = 0.5


class ChainEvolver:
    """Evolutionary search over CARL chain space.

    Parameters
    ----------
    generator : MAGEGenerator
        Used to seed the initial population.
    evolve_config : EvolveConfig
        Knobs for benchmark size, population, generations, operator weights.
    llm_client : LLMClient | None
        LLM for grading and LLM-driven operators. If ``None``, falls back to
        the generator's own client.
    rng_seed : int | None
        Seed for reproducibility.
    """

    def __init__(
        self,
        generator: MAGEGenerator,
        evolve_config: EvolveConfig | None = None,
        llm_client: LLMClient | None = None,
        rng_seed: int | None = None,
    ) -> None:
        self._gen = generator
        self._cfg = evolve_config or EvolveConfig()
        # Reach into the generator for the LLM client if one wasn't supplied.
        self._llm = llm_client or generator._llm  # noqa: SLF001 — internal coupling intentional
        self._memory = generator._memory  # noqa: SLF001
        self._mage_config: MAGEConfig = generator._config  # noqa: SLF001
        self._rng = random.Random(rng_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evolve(self, query: str) -> EvolveResult:
        """Run the full evolutionary loop for one query."""
        await self._llm.resolve_model()

        # Stage 0: domain analysis (reuses the existing agent).
        analyzer = DomainAnalyzer(self._llm)
        domain_analysis = await analyzer.analyze(query)

        # Stage 1: benchmark synthesis.
        synthesizer = BenchmarkSynthesizer(
            llm=self._llm,
            memory=self._memory,
            web_research_agent=WebResearchAgent(self._llm) if self._cfg.enable_web_synthesis else None,
        )
        benchmark = await synthesizer.synthesize(query, domain_analysis, self._cfg, self._mage_config)
        logger.info("Synthesized %d benchmark tasks", len(benchmark.tasks))

        if not benchmark.tasks:
            # No benchmark → no evolutionary signal. Fall back to a single
            # generation and return the raw output as best.
            return await self._fallback_no_benchmark(query, benchmark)

        # Stage 2: initial population.
        population = await self._build_initial_population(query, domain_analysis)

        # Stage 3: grade everyone.
        grader = ChainGrader(self._llm)
        await self._grade_population(population, benchmark, grader)
        history: list[EvolveGeneration] = [self._snapshot(population, generation=0)]

        # Stage 4-5: evolve loop.
        plateau_counter = 0
        last_best = history[0].best_score
        early_stop_reason = ""

        for gen_idx in range(1, self._cfg.num_generations + 1):
            if last_best >= self._cfg.score_threshold:
                early_stop_reason = f"score_threshold reached ({last_best:.3f})"
                break

            children, invalid_children = await self._next_generation(population, benchmark, grader, query, gen_idx)
            await self._grade_population(children, benchmark, grader)
            population = self._select(population, children)
            snap = self._snapshot(population, generation=gen_idx, invalid_children=invalid_children)
            history.append(snap)

            plateau_counter, plateau_stop = self._plateau_step(
                snap.best_score,
                last_best,
                plateau_counter,
                min_delta=self._cfg.plateau_min_delta,
                patience=self._cfg.plateau_patience,
            )
            last_best = snap.best_score

            if plateau_stop:
                early_stop_reason = f"plateau ({plateau_counter} gens without improvement)"
                break

        # Sort final population descending by score.
        ranked = sorted(population, key=self._rank_key, reverse=True)
        result = EvolveResult(
            query=query,
            best_individual=ranked[0],
            benchmark=benchmark,
            history=history,
            top_k=ranked[: max(1, self._cfg.elitism)],
            total_generations_run=len(history) - 1,
            total_individuals_evaluated=sum(g.population_size for g in history),
            early_stopped=bool(early_stop_reason),
            early_stop_reason=early_stop_reason,
        )

        # Optionally persist for future runs.
        if self._cfg.save_benchmark_to_memory and self._memory is not None:
            await self._save_benchmark_best_effort(result, domain_analysis)

        return result

    @staticmethod
    def _plateau_step(
        best_score: float, last_best: float, counter: int, *, min_delta: float, patience: int
    ) -> tuple[int, bool]:
        """Advance the plateau counter for one generation (WP-3 / T24).

        An improvement must EXCEED ``last_best`` by more than ``min_delta`` to
        reset the counter; an improvement of exactly ``min_delta`` (or less) still
        counts as a plateau. Returns ``(new_counter, should_stop)`` where
        ``should_stop`` is ``counter >= patience``. Pure + total so the early-stop
        behaviour is unit-testable on a synthetic score sequence.
        """
        if best_score <= last_best + min_delta:
            counter += 1
        else:
            counter = 0
        return counter, counter >= patience

    # ------------------------------------------------------------------
    # Stage 2 — initial population
    # ------------------------------------------------------------------

    async def _build_initial_population(self, query: str, domain_analysis: DomainAnalysis) -> list[Individual]:
        """Generate ``population_size`` diverse chains via the MAGE generator.

        WP-6: a single ``generate(query, return_candidates=True)`` carries every
        topology-sampled plan candidate through DAG → describe → validation while
        the research / domain / capability stages run **once** — so N seed chains
        share one research pass instead of N independent generations. When that
        yields fewer than ``population_size`` members (e.g. fast mode, or the
        selector found only one topology), the remaining slots are topped up with
        per-temperature single generations.

        Top-up diversity comes from per-call temperature variation; each top-up
        gets its OWN deep-copied config + generator, so concurrent generations
        don't race on a shared ``temperature`` — the evolver's shared config /
        generator is never mutated (WP-1).
        """
        base_temp = self._mage_config.temperature
        all_temps = self._cfg.initial_temperatures or [base_temp]

        async def _gen_one(temp: float) -> Individual | None:
            # Per-call config + generator: deep-copy so parallel generations
            # don't clobber each other's temperature. The shared config is left
            # untouched (previously this mutated `self._mage_config.temperature`
            # under asyncio.gather, so the actual temperature of each call was
            # decided by interleaving and diversification silently broke).
            cfg = self._mage_config.model_copy(deep=True, update={"temperature": temp})
            gen = MAGEGenerator(config=cfg, progress=self._gen._progress)  # noqa: SLF001
            try:
                result = await gen.generate(query)
            except Exception as exc:
                logger.warning("Initial population: generation failed at temp=%.2f: %s", temp, exc)
                return None
            assert not isinstance(result, list)  # no return_candidates here
            return Individual(
                individual_id=_short_id("init"),
                chain_dict=result.chain_dict,
                operator="initial",
                generation=0,
                usage=gen._llm.total_usage(),  # noqa: SLF001 — per-individual usage (own client)
            )

        # 1. Research-once: one generation, fanned out over topology candidates.
        out: list[Individual] = []
        try:
            shared = await self._gen.generate(query, return_candidates=True)
        except Exception as exc:
            logger.warning("Initial population: shared candidate generation failed: %s", exc)
            shared = []
        shared_results = shared if isinstance(shared, list) else [shared]
        shared_usage = self._gen._llm.total_usage()  # noqa: SLF001 — shared research spend
        for res in shared_results[: self._cfg.population_size]:
            out.append(
                Individual(
                    individual_id=_short_id("init"),
                    chain_dict=res.chain_dict,
                    operator="initial",
                    generation=0,
                    usage=shared_usage,
                )
            )

        # 2. Top up any remaining slots with per-temperature single generations.
        remaining = self._cfg.population_size - len(out)
        if remaining > 0:
            topup_temps = [all_temps[i % len(all_temps)] for i in range(remaining)]
            extra = await asyncio.gather(*[_gen_one(t) for t in topup_temps])
            out.extend(ind for ind in extra if ind is not None)

        if not out:
            raise RuntimeError("Failed to generate any initial population members")
        return out

    # ------------------------------------------------------------------
    # Stage 3 — grading
    # ------------------------------------------------------------------

    async def _grade_population(
        self,
        population: list[Individual],
        benchmark: BenchmarkSynthesisResult,
        grader: ChainGrader,
    ) -> None:
        async def _grade_one(ind: Individual) -> None:
            ind.score = await grader.grade(ind.chain_dict, benchmark, self._cfg)

        await asyncio.gather(*[_grade_one(i) for i in population])

    # ------------------------------------------------------------------
    # Stage 4 — produce next generation
    # ------------------------------------------------------------------

    async def _next_generation(
        self,
        population: list[Individual],
        benchmark: BenchmarkSynthesisResult,
        grader: ChainGrader,
        query: str,
        gen_idx: int,
    ) -> tuple[list[Individual], int]:
        """Build a child population the same size as the parent population.

        Returns ``(children, invalid_children)``. A slot whose operator yields an
        invalid (or over-budget) child is retried up to twice with a *different*
        operator (WP-3), so one flaky operator no longer silently shrinks the
        generation; the count of rejected children rides into the snapshot.
        """
        target_size = max(1, self._cfg.population_size - self._cfg.elitism)
        children: list[Individual] = []
        invalid_children = 0

        ranked_parents = sorted(population, key=self._rank_key, reverse=True)

        for _ in range(target_size):
            child: Individual | None = None
            tried: set[str] = set()
            for _attempt in range(3):  # 1 try + up to 2 retries with a different op
                op = self._pick_operator(exclude=tried)
                tried.add(op)
                try:
                    child = await self._apply_operator(op, ranked_parents, query)
                except Exception as exc:
                    logger.warning("Operator %s failed: %s", op, exc)
                    child = None
                if child is not None:
                    break
                invalid_children += 1
            if child is not None:
                child.generation = gen_idx
                children.append(child)

        return children, invalid_children

    def _pick_operator(self, exclude: set[str] | None = None) -> str:
        weights = {
            "mutate_aim": self._cfg.weight_mutate_aim,
            "change_step_type": self._cfg.weight_change_step_type,
            "insert_step": self._cfg.weight_insert_step,
            "remove_step": self._cfg.weight_remove_step,
            "swap_steps": self._cfg.weight_swap_steps,
            "crossover": self._cfg.weight_crossover,
            "llm_repair": self._cfg.weight_llm_repair,
            "reflective_mutate": self._cfg.weight_reflective_mutate,
        }
        if exclude:
            for name in exclude:
                weights.pop(name, None)
        names = list(weights.keys())
        if not names:
            return "mutate_aim"
        ws = [max(0.0, w) for w in weights.values()]
        if sum(ws) <= 0:
            return names[0]
        return self._rng.choices(names, weights=ws, k=1)[0]

    async def _apply_operator(self, op: str, ranked: list[Individual], query: str) -> Individual | None:
        if not ranked:
            return None

        # Bias parent selection toward the top half — soft elitism within the operator.
        upper = max(1, len(ranked) // 2)
        parent_a = self._rng.choice(ranked[:upper])
        parent_b = self._rng.choice(ranked) if op == "crossover" else parent_a

        if op == "mutate_aim":
            child_dict = mutate_aim(parent_a.chain_dict, self._rng)
        elif op == "change_step_type":
            child_dict = change_step_type(parent_a.chain_dict, self._rng)
        elif op == "insert_step":
            child_dict = insert_step(parent_a.chain_dict, self._rng)
        elif op == "remove_step":
            child_dict = remove_step(parent_a.chain_dict, self._rng)
        elif op == "swap_steps":
            child_dict = swap_steps(parent_a.chain_dict, self._rng)
        elif op == "crossover":
            child_dict = crossover(parent_a.chain_dict, parent_b.chain_dict, self._rng)
        elif op == "llm_repair":
            child_dict = await llm_repair(parent_a.chain_dict, parent_a.score, query, self._llm)
        elif op == "reflective_mutate":
            child_dict = await reflective_mutate(parent_a.chain_dict, parent_a.score, query, self._llm)
        else:
            return None

        # Validation: skip children that produced an invalid chain. Try
        # CARL-compat first to catch most issues.
        try:
            child_dict = to_carl_compat(child_dict)
            validate_carl_json(child_dict, strict_carl=True)
        except Exception as exc:
            logger.debug("Operator %s produced invalid chain: %s", op, exc)
            return None

        # Cost budget guard — reject mutants that exceed configured limits.
        if self._cfg.max_cost_usd is not None or self._cfg.max_tokens_per_chain is not None:
            fits, estimate = fits_in_budget(
                child_dict,
                max_cost_usd=self._cfg.max_cost_usd,
                max_total_tokens=self._cfg.max_tokens_per_chain,
                pricing=self._cfg.pricing or None,
                default_model=self._mage_config.model,
            )
            if not fits:
                logger.debug(
                    "Operator %s mutant over budget: $%.4f / %d tokens — skipped",
                    op,
                    estimate.total_cost_usd,
                    estimate.total_tokens,
                )
                return None

        return Individual(
            individual_id=_short_id(op),
            chain_dict=child_dict,
            parents=[parent_a.individual_id, parent_b.individual_id] if op == "crossover" else [parent_a.individual_id],
            operator=op,
        )

    # ------------------------------------------------------------------
    # Selection (elitism + best-overall, gaming-penalized)
    # ------------------------------------------------------------------

    @staticmethod
    def _rank_key(ind: Individual) -> float:
        """Survival/selection ranking score: the judged ``overall`` minus
        :data:`_GAMING_PENALTY` when the judge flagged the chain as suspected of
        gaming the benchmark (P-5/T25), so gamed chains sink below honest ones of
        comparable raw score. Honest chains rank exactly by ``overall``.
        """
        return ind.score.overall - (_GAMING_PENALTY if ind.score.suspect_gaming else 0.0)

    def _select(self, parents: list[Individual], children: list[Individual]) -> list[Individual]:
        merged = parents + children
        if not self._cfg.enable_pareto_selection:
            merged.sort(key=self._rank_key, reverse=True)
            # Keep population_size; first elitism slots are guaranteed to be the
            # best parents anyway because we never mutate them in place.
            return merged[: self._cfg.population_size]
        return self._select_pareto(merged)

    def _individual_cost(self, ind: Individual) -> float:
        """Cost axis for Pareto selection (P1.8): estimated total tokens (pricing-free)."""
        try:
            return float(estimate_chain_cost(ind.chain_dict, default_model=self._mage_config.model).total_tokens)
        except Exception:
            return 0.0

    def _select_pareto(self, merged: list[Individual]) -> list[Individual]:
        """GEPA-style survivor selection over (quality ↑, cost ↓) (plan item P1.8).

        Take the Pareto-non-dominated front first (cheaper-or-better chains never
        crowded out by a pricier equal), best-quality first; then fill the rest by
        quality. The global best is always on the front, so elitism is preserved.
        """
        costs = {id(i): self._individual_cost(i) for i in merged}
        front = _pareto_front(merged, quality=self._rank_key, cost=lambda i: costs[id(i)])
        front.sort(key=self._rank_key, reverse=True)
        size = self._cfg.population_size
        if len(front) >= size:
            return front[:size]
        front_ids = {id(i) for i in front}
        rest = sorted((i for i in merged if id(i) not in front_ids), key=self._rank_key, reverse=True)
        return front + rest[: size - len(front)]

    # ------------------------------------------------------------------
    # Snapshot / fallback / persistence
    # ------------------------------------------------------------------

    def _snapshot(
        self, population: list[Individual], *, generation: int, invalid_children: int = 0
    ) -> EvolveGeneration:
        scores = sorted([i.score.overall for i in population], reverse=True)
        best_ind = max(population, key=self._rank_key)
        op_counts: dict[str, int] = {}
        for ind in population:
            op_counts[ind.operator] = op_counts.get(ind.operator, 0) + 1
        median = scores[len(scores) // 2] if scores else 0.0
        return EvolveGeneration(
            generation=generation,
            population_size=len(population),
            best_score=best_ind.score.overall if population else 0.0,
            median_score=median,
            best_individual_id=best_ind.individual_id,
            operators_used=op_counts,
            invalid_children=invalid_children,
        )

    async def _fallback_no_benchmark(self, query: str, benchmark: BenchmarkSynthesisResult) -> EvolveResult:
        result = await self._gen.generate(query)
        ind = Individual(
            individual_id=_short_id("init"),
            chain_dict=result.chain_dict,
            operator="initial",
        )
        return EvolveResult(
            query=query,
            best_individual=ind,
            benchmark=benchmark,
            top_k=[ind],
            total_individuals_evaluated=1,
            early_stopped=True,
            early_stop_reason="no benchmark synthesized",
        )

    async def _save_benchmark_best_effort(self, result: EvolveResult, domain: DomainAnalysis) -> None:
        if self._memory is None:
            return
        try:
            await self._memory.save_benchmark(
                domain=domain.domain,
                tasks=[t.model_dump() for t in result.benchmark.tasks],
                best_chain=result.best_individual.chain_dict,
                best_score=result.best_individual.score.overall,
                query=result.query,
            )
        except Exception as exc:
            logger.debug("save_benchmark_best_effort: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _pareto_front(items: list, quality, cost) -> list:  # noqa: ANN001
    """Return the (quality ↑, cost ↓) Pareto-non-dominated subset (plan item P1.8).

    ``a`` dominates ``b`` when ``a`` is no worse on both axes (quality ≥, cost ≤)
    and strictly better on at least one. Items not dominated by any other survive.
    Stable: returns survivors in their original order.
    """
    out = []
    for a in items:
        qa, ca = quality(a), cost(a)
        dominated = any(
            (quality(b) >= qa and cost(b) <= ca and (quality(b) > qa or cost(b) < ca)) for b in items if b is not a
        )
        if not dominated:
            out.append(a)
    return out


def deepcopy_chain(chain: dict[str, Any]) -> dict[str, Any]:  # tiny export for tests
    return copy.deepcopy(chain)
