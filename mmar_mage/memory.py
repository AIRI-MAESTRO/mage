"""
Memory integration layer for MAGE.

Provides a thin wrapper around ``gigaevo-memory`` :class:`MemoryClient`
that gracefully degrades when the memory service is unavailable or the
package is not installed.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from .exceptions import MAGEMemoryError
from .schemas import MAGEConfig, MemoryHit

logger = logging.getLogger(__name__)


def is_reusable_step(step: dict[str, Any]) -> bool:
    """Decide whether a CARL step is reusable enough to save as a template (§5 P1).

    "Reusable" = an ``llm`` step whose prompt parts (``aim``,
    ``reasoning_questions``, ``stage_action``, ``example_reasoning``)
    contain no dynamic placeholders that bind it to this specific
    chain. Dynamic placeholders include:

    - ``{{...}}`` Jinja-style template variables
    - ``step_N_output`` / ``step_N_result`` substitutions
    - ``{output_of_step_*}`` substitutions

    Conservative by design — false negatives cost a missed save, but
    false positives clutter Memory with steps that won't generalise.

    Parameters
    ----------
    step : dict[str, Any]
        CARL step dict.

    Returns
    -------
    bool
        ``True`` iff the step is safe to ingest as a template.
    """
    if not isinstance(step, dict):
        return False
    if (step.get("step_type") or "llm") != "llm":
        return False

    # Minimum viable content: aim must be non-empty and meaningful.
    aim = (step.get("aim") or "").strip()
    if len(aim) < 8:
        return False

    haystack = " ".join(
        str(step.get(k) or "") for k in ("aim", "reasoning_questions", "stage_action", "example_reasoning")
    )
    dynamic_patterns = ("{{", "}}", "step_", "{output_of_step", "_output", "_result")
    if any(p in haystack for p in dynamic_patterns):
        return False

    return True


def _merge_care_chain_metadata(
    chain_dict: dict[str, Any],
    *,
    task_description: str | None,
    context_files: list[dict[str, Any]] | None,
    mage_metadata: dict[str, Any] | None,
    display_name: str | None,
    description: str | None,
) -> dict[str, Any]:
    """Merge a `CareChainMetadata` block into a CARL chain dict.

    Returns a new chain dict with a ``metadata`` block containing the
    CARE-shaped fields. Used by :meth:`MemoryManager.save_chain` so CARE's
    "Re-run from library" can re-prime ``ReasoningContext`` via CARL's
    ``ReasoningContext.from_chain_inputs()``.

    Prefers the typed helper ``CareChainMetadata.merge_into_content()`` from
    ``gigaevo-client`` when that package is installed; falls back to a
    direct dict-merge that mirrors the same shape when it isn't, so this
    module's only hard dep remains the (already-optional) ``gigaevo-memory``.

    Skips the merge entirely (returns the input unchanged) when every
    field is ``None`` — keeps backwards compatibility for callers that
    only pass the chain and domain.
    """
    if all(v is None for v in (task_description, context_files, mage_metadata, display_name, description)):
        return chain_dict

    # Try the typed helper first.
    try:
        from gigaevo_client.models import CareChainMetadata, ContextFileRef

        meta = CareChainMetadata(
            task_description=task_description or "",
            context_files=[ContextFileRef(**cf) for cf in (context_files or [])],
            generated_by="mage",
            mage_metadata=mage_metadata or {},
            display_name=display_name or "",
            description=description or "",
        )
        merged = meta.merge_into_content(chain_dict)
        if isinstance(merged, dict):
            return merged
        logger.warning(
            "CareChainMetadata.merge_into_content() returned non-dict %s; falling back to direct dict merge",
            type(merged).__name__,
        )
    except ImportError:
        logger.debug("gigaevo_client not installed; using direct dict merge for CARE metadata")
    except Exception as exc:
        logger.warning("CareChainMetadata helper failed (%s); falling back to direct dict merge", exc)

    # Direct dict-merge fallback — same shape as the typed helper produces.
    merged_dict: dict[str, Any] = dict(chain_dict)
    existing = merged_dict.get("metadata")
    care_block: dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
    if task_description is not None:
        care_block["task_description"] = task_description
    if context_files is not None:
        care_block["context_files"] = list(context_files)
    care_block["generated_by"] = "mage"
    if mage_metadata is not None:
        care_block["mage_metadata"] = mage_metadata
    if display_name is not None:
        care_block["display_name"] = display_name
    if description is not None:
        care_block["description"] = description
    merged_dict["metadata"] = care_block
    return merged_dict


def _memory_sdk() -> Any:
    """Return the memory SDK module — the renamed ``gigaevo_client``
    (0.3.0+), falling back to the legacy ``gigaevo_memory`` shim.

    Centralises the dual-package lookup so every call site
    (``MemoryClient`` / ``SearchType``) resolves the same way and works
    whether the deployment ships the new client (the normal CARE
    runtime) or the old shim. Raises ``ImportError`` only when neither
    is installed (callers already guard on ``_available``)."""
    try:
        import gigaevo_client as _sdk
    except ImportError:
        import gigaevo_memory as _sdk
    return _sdk


# Strong execution-error signals — used to spot a tool step that returned an error
# STRING while reporting success=true (the R6 "error as data" masking).
_TOOL_ERROR_MARKERS: tuple[str, ...] = (
    "invalid idna",
    "traceback (most recent call",
    "exception",
    "errno",
    "connection refused",
    "connection error",
    "timed out",
    "timeout",
    "invalid url",
    "no address associated with hostname",
    "nodename nor servname",
    "failed to resolve",
    "name or service not known",
    "max retries exceeded",
)


def _looks_like_tool_error(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _TOOL_ERROR_MARKERS)


def find_masked_tool_errors(step_results: list[Any] | None) -> list[dict[str, Any]]:
    """Scan a run's executed steps for tool failures, INCLUDING the "error as data
    with success=true" masking (R6/T34). Returns one record per failing step —
    ``{number, tool_name, error, masked}``. A ``masked`` error (a tool step that
    reported success but whose output reads like an error) is the run-096 class
    that slipped through as ``status: success``.
    """
    failures: list[dict[str, Any]] = []
    for s in step_results or []:
        if not isinstance(s, dict):
            continue
        is_tool = str(s.get("step_type") or "").lower() in ("tool", "mcp")
        err = s.get("error")
        success = s.get("success")
        raw_out = s.get("output") if s.get("output") is not None else s.get("result")
        out_text = (
            raw_out
            if isinstance(raw_out, str)
            else (json.dumps(raw_out, ensure_ascii=False, default=str) if raw_out is not None else "")
        )
        masked = success is True and is_tool and _looks_like_tool_error(out_text)
        if not (success is False or err or masked):
            continue
        failures.append(
            {
                "number": s.get("number"),
                "tool_name": s.get("tool_name") or s.get("name"),
                "error": str(err) if err else (out_text[:200] or "step reported failure"),
                "masked": bool(masked),
            }
        )
    return failures


def extract_run_facts(step_results: list[Any] | None, *, max_chars: int = 500) -> dict[str, str]:
    """Pull a run's SUCCESSFUL tool / mcp step outputs into a compact facts dict (T17),
    so CARE can persist them to session memory via :meth:`MemoryManager.save_run_facts`
    and a follow-up question reuses them instead of cold-starting — e.g. the IP one
    step fetched, the country a geolocation returned (run 773 re-derived the IP from
    the dialogue text because nothing was written back). Keyed by each step's output
    key (or ``step_<N>``); values are the truncated string output. Failed steps and
    non-tool steps are skipped (their results aren't reusable facts)."""
    facts: dict[str, str] = {}
    for s in step_results or []:
        if not isinstance(s, dict):
            continue
        if str(s.get("step_type") or "").lower() not in ("tool", "mcp"):
            continue
        if s.get("success") is False or s.get("error"):
            continue
        raw = s.get("output") if s.get("output") is not None else s.get("result")
        if raw is None:
            continue
        text = (raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)).strip()
        if not text:
            continue
        key = str(s.get("tool_output_key") or s.get("output_key") or f"step_{s.get('number')}")
        facts[key] = text[:max_chars]
    return facts


class MemoryManager:
    """Manages interactions with the gigaevo-memory service.

    Parameters
    ----------
    config : MAGEConfig
        MAGE configuration (``memory_base_url``, ``memory_tags``, ``enable_memory``).

    Notes
    -----
    All public methods return ``None`` on failure rather than raising,
    ensuring that memory issues never block chain generation.

    Examples
    --------
    >>> mgr = MemoryManager(MAGEConfig(enable_memory=True))
    >>> key = await mgr.save_chain(chain_dict, domain="finance")
    """

    def __init__(self, config: MAGEConfig) -> None:
        self._config = config
        self._available = self._check_availability()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _check_availability() -> bool:
        """Return ``True`` if the memory SDK is importable.

        Accepts the renamed ``gigaevo_client`` (0.3.0+) or the legacy
        ``gigaevo_memory`` shim."""
        for pkg in ("gigaevo_client", "gigaevo_memory"):
            try:
                __import__(pkg)
                return True
            except ImportError:
                continue
        return False

    def _search_params(self, query: str, entity_type: str, top_k: int) -> tuple[dict[str, Any], Any]:
        """Build gigaevo search kwargs + the resolved ``SearchType`` from config (P1.18).

        Centralizes search-type / hybrid-weight / namespace selection so every
        recall path shares one configured strategy. Unknown modes fall back to
        BM25.
        """
        SearchType = _memory_sdk().SearchType  # noqa: N806 (SDK enum)

        mode = (self._config.memory_search_mode or "bm25").lower()
        search_type = getattr(SearchType, mode.upper(), SearchType.BM25)

        kwargs: dict[str, Any] = {
            "query": query,
            "search_type": search_type,
            "entity_type": entity_type,
            "top_k": top_k,
        }
        if mode == "hybrid":
            kwargs["hybrid_weights"] = tuple(self._config.memory_hybrid_weights)
        if self._config.memory_namespace:
            kwargs["namespace"] = self._config.memory_namespace
        return kwargs, search_type

    def _run_search(self, search_fn: Any, query: str, *, entity_type: str, top_k: int) -> list[Any]:
        """Invoke a gigaevo search callable with the configured mode + BM25 fallback (P1.18).

        If the server rejects a ``vector``/``hybrid`` request (e.g. vector search
        disabled), the call is retried once as BM25 so recall degrades instead
        of failing.
        """
        SearchType = _memory_sdk().SearchType  # noqa: N806 (SDK enum)

        kwargs, search_type = self._search_params(query, entity_type, top_k)
        try:
            return search_fn(**kwargs)
        except Exception as exc:
            if search_type == SearchType.BM25:
                raise
            logger.warning("MemoryManager: %s search rejected (%s); retrying with BM25", search_type, exc)
            kwargs["search_type"] = SearchType.BM25
            kwargs.pop("hybrid_weights", None)
            return search_fn(**kwargs)

    def _search(self, client: Any, query: str, *, entity_type: str, top_k: int) -> list[Any]:
        """Entity search via ``client.search`` (returns typed specs). See :meth:`_run_search`."""
        return self._run_search(client.search, query, entity_type=entity_type, top_k=top_k)

    def _search_hits(self, client: Any, query: str, *, entity_type: str, top_k: int) -> list[Any]:
        """Scored search via ``client.search_hits`` → ``list[SearchHitData]`` (P1.18).

        Each hit exposes the raw ``.content`` dict, ``.entity_id`` and a relevance
        ``.score`` — unlike the typed ``search`` path, which validates into
        ``MemoryCardSpec``/specs (dropping arbitrary keys and the score). Use this
        whenever the caller needs scores or the verbatim stored payload.
        """
        return self._run_search(client.search_hits, query, entity_type=entity_type, top_k=top_k)

    def _search_cards(self, client: Any, query: str, *, top_k: int) -> list[Any]:
        """Search memory cards via ``search_hits``, returning raw ``SearchHitData`` (P1.18 item 3)."""
        return self._search_hits(client, query, entity_type="memory_card", top_k=top_k)

    def _save_card(
        self,
        client: Any,
        content: dict[str, Any],
        *,
        name: str,
        tags: list[str],
        when_to_use: str | None = None,
    ) -> str | None:
        """Persist a reusable-knowledge payload as a gigaevo memory card (P1.18 item 3).

        ``content`` is stored verbatim (``save_memory_card`` accepts a raw dict);
        callers should include a ``"kind"`` key so :meth:`_search_cards` consumers
        can filter by type. Returns the new ``entity_id`` (or ``None``).
        """
        ref = client.save_memory_card(
            memory_card=content,
            name=name,
            tags=tags,
            when_to_use=when_to_use,
            namespace=self._config.memory_namespace,
        )
        return getattr(ref, "entity_id", None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def save_chain(
        self,
        chain_dict: dict[str, Any],
        *,
        domain: str = "general",
        name: str | None = None,
        task_description: str | None = None,
        context_files: list[dict[str, Any]] | None = None,
        mage_metadata: dict[str, Any] | None = None,
        display_name: str | None = None,
        description: str | None = None,
        evolution_meta: dict[str, Any] | None = None,
        entity_id: str | None = None,
        channel: str = "latest",
    ) -> str | None:
        """Save a generated chain to gigaevo-memory.

        Parameters
        ----------
        chain_dict : dict[str, Any]
            CARL chain as a Python dict.
        domain : str
            Domain label used for naming and tagging.
        name : str | None
            Explicit chain name; auto-generated if ``None``.
        task_description : str | None
            Original NL query that produced this chain. Persisted under the
            chain's ``metadata`` block so CARE's "Re-run from library" can
            re-prime the ``ReasoningContext`` via
            ``ReasoningContext.from_chain_inputs()``.
        context_files : list[dict] | None
            File references the user attached when running the chain
            (``{path, sha256, size_bytes}``). Persisted alongside
            ``task_description`` for re-run.
        mage_metadata : dict | None
            Full ``MAGEMetadata.model_dump()`` so the library row records
            which model/stages produced the chain.
        display_name, description : str | None
            CARE library-ready naming (from ``MAGEMetadata.suggested_*``).
            Surfaced in the library list view.
        evolution_meta : dict | None
            §5 P2 quality signals (``step_critique_score``,
            ``verification_passed``, ``refine_quality_delta``,
            ``mcts_best_reward``, ...) — the platform's evolution
            engine reads these off the chain entity as fitness priors.
            Persisted under ``chain.metadata.evolution_meta``.
        entity_id : str | None
            When provided, the chain is saved as a NEW VERSION under this
            existing entity (the SDK records a new version on the same id)
            instead of creating a fresh entity. Used by the edit flow
            (:meth:`MAGEGenerator.edit`). Default ``None`` ⇒ new entity.
        channel : str
            Version channel to write when ``entity_id`` is given (default
            ``"latest"``). Ignored for fresh saves.

        Returns
        -------
        str | None
            The ``entity_id`` (memory key) on success, or ``None`` on failure.
        """
        if not self._config.enable_memory:
            logger.debug("MemoryManager: memory disabled, skipping save")
            return None

        if not self._available:
            logger.warning("MemoryManager: gigaevo-memory not installed, skipping save")
            return None

        chain_to_save = _merge_care_chain_metadata(
            chain_dict,
            task_description=task_description,
            context_files=context_files,
            mage_metadata=mage_metadata,
            display_name=display_name,
            description=description,
        )
        if evolution_meta:
            # Always nest under `metadata.evolution_meta` so the chain
            # row is self-describing for the evolution engine. Honour
            # any pre-existing `metadata` block the merge produced.
            md = chain_to_save.setdefault("metadata", {})
            if not isinstance(md, dict):
                logger.warning("save_chain: existing `metadata` is not a dict; replacing")
                md = {}
                chain_to_save["metadata"] = md
            md["evolution_meta"] = dict(evolution_meta)

        # The Memory server validates the full CARL chain shape — a
        # top-level ``version`` and ``metadata`` are required. MAGE's raw
        # generated dict omits ``version`` (CARE-saved ReasoningChains
        # carry it), so inject defaults to avoid a 400 rejection on save.
        chain_to_save.setdefault("version", "1.0")
        if not isinstance(chain_to_save.get("metadata"), dict):
            chain_to_save["metadata"] = {}

        try:
            return self._do_save(
                chain_to_save,
                domain=domain,
                name=name,
                entity_id=entity_id,
                channel=channel,
            )
        except Exception as exc:
            logger.error("MemoryManager: failed to save chain: %s", exc)
            return None

    def _do_save(
        self,
        chain_dict: dict[str, Any],
        *,
        domain: str,
        name: str | None,
        entity_id: str | None = None,
        channel: str = "latest",
    ) -> str:
        """Perform the actual save via gigaevo-memory (sync).

        Raises
        ------
        MAGEMemoryError
            On any communication or serialization failure.
        """
        MemoryClient = _memory_sdk().MemoryClient  # noqa: N806 (SDK class)

        chain_name = name or f"mage_{domain}"
        tags = [domain] + self._config.memory_tags

        save_kwargs: dict[str, Any] = {"chain": chain_dict, "name": chain_name, "tags": tags}
        if entity_id:
            # Same entity_id → the SDK records a NEW VERSION under it. Only pass
            # these when versioning so the legacy gigaevo-memory shim (which may
            # not accept them) keeps working for fresh saves.
            save_kwargs["entity_id"] = entity_id
            save_kwargs["channel"] = channel

        try:
            with MemoryClient(base_url=self._config.memory_base_url) as client:
                ref = client.save_chain(**save_kwargs)
                saved_id: str = ref.entity_id
                logger.info(
                    "MemoryManager: saved chain as %s%s",
                    saved_id,
                    " (new version)" if entity_id else "",
                )
                return saved_id
        except Exception as exc:
            raise MAGEMemoryError(f"Failed to save chain to memory: {exc}") from exc

    async def load_chain(self, entity_id: str, *, channel: str = "latest") -> dict[str, Any] | None:
        """Load a saved chain's content dict by ``entity_id`` (best-effort).

        Mirrors :meth:`search_similar`'s client pattern. Returns the stored CARL
        chain dict, or ``None`` when memory is disabled, the SDK is missing, the
        id is unknown, or the SDK lacks ``get_chain_dict`` (legacy shim).

        Parameters
        ----------
        entity_id : str
            The chain entity to load.
        channel : str
            Version channel to read (``"latest"`` / ``"stable"`` / custom).
        """
        if not self._config.enable_memory or not self._available:
            return None

        try:
            MemoryClient = _memory_sdk().MemoryClient  # noqa: N806 (SDK class)

            with MemoryClient(base_url=self._config.memory_base_url) as client:
                if not hasattr(client, "get_chain_dict"):
                    logger.warning("MemoryManager: SDK lacks get_chain_dict; cannot load_chain")
                    return None
                content = client.get_chain_dict(entity_id, channel=channel)
                return content if isinstance(content, dict) else None
        except Exception as exc:
            logger.warning("MemoryManager: load_chain(%s) failed: %s", entity_id, exc)
            return None

    async def search_similar(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Search for similar chains in memory (best-effort).

        Parameters
        ----------
        query : str
            Search query.
        top_k : int
            Maximum number of results.

        Returns
        -------
        list[dict[str, Any]]
            Matching chain metadata, or empty list on failure.
        """
        if not self._config.enable_memory or not self._available:
            return []

        try:
            MemoryClient = _memory_sdk().MemoryClient  # noqa: N806 (SDK class)

            with MemoryClient(base_url=self._config.memory_base_url) as client:
                # Use search_hits (generic scored search across entity types);
                # client.search is memory-card-specialised in gigaevo_client
                # 0.3.0 and returns specs without `entity_id`.
                results = self._search_hits(client, query, entity_type="chain", top_k=top_k)
                return [
                    {
                        "entity_id": r.entity_id,
                        "name": getattr(r, "name", ""),
                        "score": float(getattr(r, "score", 0.0) or 0.0),
                    }
                    for r in results
                ]
        except Exception as exc:
            logger.warning("MemoryManager: search failed: %s", exc)
            return []

    async def iterative_search(
        self,
        query: str,
        domain_concepts: list[str],
        sub_queries_count: int = 3,
        top_k: int = 5,
        threshold: float = 0.4,
        extra_queries: list[str] | None = None,
    ) -> list[MemoryHit]:
        """Perform iterative multi-query recall of past chains from memory.

        Expands the query with domain concepts into sub-queries, runs the
        configured search (hybrid by default) for each over a single client
        connection, deduplicates by entity, filters by relevance ``threshold``,
        and returns hits sorted best-first.

        Uses ``search_hits`` so each result carries a real relevance ``score``.
        (The previous implementation read a ``score`` that the underlying call
        never returned, so every hit scored 0.0 and was filtered out — i.e. it
        always returned ``[]``; P1.18 item 4 fix.) LLM-generated sub-queries and
        an applicability gate are layered on at the agent level in P1.1.

        Parameters
        ----------
        query : str
            Original user query.
        domain_concepts : list[str]
            Key domain concepts from domain analysis.
        sub_queries_count : int
            Number of sub-queries to generate.
        top_k : int
            Max results per sub-query.
        threshold : float
            Minimum relevance score for results.

        Returns
        -------
        list[MemoryHit]
            Deduplicated, threshold-filtered memory hits, sorted by score desc.
        """
        if not self._config.enable_memory or not self._available:
            return []

        # ``extra_queries`` (e.g. P1.1 second-wave gap queries) are searched
        # directly; otherwise build sub-queries from the query + domain concepts.
        if extra_queries:
            sub_queries = list(extra_queries)
        else:
            # Short, per-concept sub-queries recall FAR better than long
            # strings: the Memory BM25 index ANDs query terms, so a long query
            # that includes any non-indexed word matches nothing. Search each
            # salient concept on its own, with the full query as a fallback.
            concepts = [c.strip() for c in domain_concepts if c and c.strip()]
            sub_queries = concepts[: max(0, sub_queries_count - 1)] + [query]
            sub_queries = list(dict.fromkeys(sub_queries))  # de-dupe, keep order

        seen_ids: set[str] = set()
        hits: list[MemoryHit] = []

        try:
            MemoryClient = _memory_sdk().MemoryClient  # noqa: N806 (SDK class)

            with MemoryClient(base_url=self._config.memory_base_url) as client:
                for sq in sub_queries:
                    try:
                        raw_hits = self._search_hits(client, sq, entity_type="chain", top_k=top_k)
                    except Exception as exc:
                        logger.warning("MemoryManager: iterative_search sub-query failed: %s", exc)
                        continue
                    for h in raw_hits:
                        eid = getattr(h, "entity_id", "") or ""
                        if not eid or eid in seen_ids:
                            continue
                        seen_ids.add(eid)
                        score = float(getattr(h, "score", 0.0) or 0.0)
                        if score < threshold:
                            continue
                        content = getattr(h, "content", None) or {}
                        hits.append(
                            MemoryHit(
                                entity_id=eid,
                                name=getattr(h, "name", "") or content.get("name", ""),
                                score=score,
                                abstract_structure=content,
                            )
                        )
        except Exception as exc:
            logger.warning("MemoryManager: iterative_search failed: %s", exc)
            return []

        hits.sort(key=lambda h: h.score, reverse=True)

        return hits

    async def get_capabilities(self, environment_id: str = "default") -> dict[str, Any] | None:
        """Query gigaevo-memory for CapabilityRegistry entity.

        Parameters
        ----------
        environment_id : str
            Environment identifier to look up.

        Returns
        -------
        dict[str, Any] | None
            Capability registry data, or ``None`` if not found or unavailable.
        """
        if not self._config.enable_memory or not self._available:
            return None

        try:
            MemoryClient = _memory_sdk().MemoryClient  # noqa: N806 (SDK class)

            with MemoryClient(base_url=self._config.memory_base_url) as client:
                results = self._search(
                    client,
                    f"CapabilityRegistry {environment_id}",
                    entity_type="capability_registry",
                    top_k=1,
                )
                if results:
                    first = results[0]
                    return {
                        "tools": getattr(first, "tools", []),
                        "mcp_servers": getattr(first, "mcp_servers", []),
                    }
                return None
        except Exception as exc:
            logger.warning("MemoryManager: get_capabilities failed: %s", exc)
            return None

    async def search_templates(
        self,
        domain: str,
        task_type: str,
        complexity: str,
    ) -> list[dict[str, Any]]:
        """Search gigaevo-memory for ChainTemplate entities.

        Parameters
        ----------
        domain : str
            Knowledge domain.
        task_type : str
            Task type.
        complexity : str
            Complexity level.

        Returns
        -------
        list[dict[str, Any]]
            Matching template metadata, or empty list on failure.
        """
        if not self._config.enable_memory or not self._available:
            return []

        try:
            MemoryClient = _memory_sdk().MemoryClient  # noqa: N806 (SDK class)

            with MemoryClient(base_url=self._config.memory_base_url) as client:
                # P1.18 item 3: templates live as memory cards (entity_type
                # "chain_template" does not exist in the server). search_hits
                # exposes the raw card content so the custom payload survives.
                hits = self._search_cards(
                    client,
                    f"chain template {domain} {task_type} {complexity}",
                    top_k=5,
                )
                return [
                    {
                        "entity_id": getattr(h, "entity_id", ""),
                        "abstract_dag": c.get("abstract_dag", []),
                        "step_type_sequence": c.get("step_type_sequence", []),
                        "quality_score": c.get("quality_score", 0.0),
                        "success_rate": c.get("success_rate", 0.0),
                        "score": getattr(h, "score", 0.0),
                    }
                    for h in hits
                    if (c := getattr(h, "content", None) or {}).get("kind") == "mage_template"
                ]
        except Exception as exc:
            logger.warning("MemoryManager: search_templates failed: %s", exc)
            return []

    async def submit_feedback(self, chain_entity_id: str, feedback: dict[str, Any]) -> None:
        """Save chain execution feedback to gigaevo-memory.

        Parameters
        ----------
        chain_entity_id : str
            Entity ID of the chain in memory.
        feedback : dict[str, Any]
            Feedback data (quality_score, success, etc.).
        """
        if not self._config.enable_memory or not self._available:
            return

        try:
            MemoryClient = _memory_sdk().MemoryClient  # noqa: N806 (SDK class)

            with MemoryClient(base_url=self._config.memory_base_url) as client:
                # P1.18 item 3: feedback persists as a typed memory card; the
                # feedback fields are flattened into the content so recall can
                # read quality_score / success / failure_reason directly.
                self._save_card(
                    client,
                    {"kind": "mage_feedback", "chain_entity_id": chain_entity_id, **feedback},
                    name=f"mage_feedback_{chain_entity_id}",
                    tags=["mage-feedback"],
                    when_to_use="Past execution outcome for a generated chain.",
                )
                logger.info("MemoryManager: saved feedback for %s", chain_entity_id)
        except Exception as exc:
            logger.warning("MemoryManager: submit_feedback failed: %s", exc)

    async def save_template(
        self,
        chain_dict: dict[str, Any],
        domain: str,
        task_type: str,
        quality_score: float,
    ) -> str | None:
        """Save a chain as a reusable template.

        Parameters
        ----------
        chain_dict : dict[str, Any]
            CARL chain dictionary.
        domain : str
            Domain label.
        task_type : str
            Task type.
        quality_score : float
            Quality score for the template.

        Returns
        -------
        str | None
            The entity_id on success, or ``None`` on failure.
        """
        if not self._config.enable_memory or not self._available:
            return None

        try:
            MemoryClient = _memory_sdk().MemoryClient  # noqa: N806 (SDK class)

            steps = chain_dict.get("steps", [])
            step_types = [s.get("step_type", "llm") for s in steps]

            with MemoryClient(base_url=self._config.memory_base_url) as client:
                # P1.18 item 3: persist as a typed memory card (not an overloaded
                # chain). The ``kind`` key lets search_templates filter on recall.
                entity_id = self._save_card(
                    client,
                    {
                        "kind": "mage_template",
                        "abstract_dag": steps,
                        "step_type_sequence": step_types,
                        "domain": domain,
                        "task_type": task_type,
                        "quality_score": quality_score,
                    },
                    name=f"mage_template_{domain}_{task_type}",
                    tags=[domain, task_type, "mage-template"],
                    when_to_use=f"Reuse for {task_type} tasks in the {domain} domain.",
                )
                if entity_id:
                    logger.info("MemoryManager: saved template as %s", entity_id)
                return entity_id
        except Exception as exc:
            logger.warning("MemoryManager: save_template failed: %s", exc)
            return None

    async def save_run_facts(
        self,
        question: str,
        facts: dict[str, Any],
        *,
        ttl_seconds: int = 3600,
    ) -> str | None:
        """Persist facts extracted from a successful run, scoped to the current
        session (T33), so a later question in the SAME session reuses them
        instead of cold-starting. No-op unless memory is enabled + available and
        a ``memory_session_id`` is configured, or when ``facts`` is empty.

        Parameters
        ----------
        question : str
            The user question the facts answer (provenance + recall display).
        facts : dict[str, Any]
            Extracted facts, e.g. ``{"ip": "185.x", "country": "Sweden"}``.
        ttl_seconds : int
            Session lifetime; recall drops cards whose ``expires_at`` has passed.

        Returns
        -------
        str | None
            The persisted ``entity_id``, or ``None`` when nothing was saved.
        """
        session_id = self._config.memory_session_id
        if not (self._config.enable_memory and self._available and session_id) or not facts:
            return None
        try:
            MemoryClient = _memory_sdk().MemoryClient  # noqa: N806 (SDK class)
            with MemoryClient(base_url=self._config.memory_base_url) as client:
                entity_id = self._save_card(
                    client,
                    {
                        "kind": "run_facts",
                        "session_id": session_id,
                        "question": question,
                        "facts": facts,
                        "expires_at": time.time() + max(0, ttl_seconds),
                    },
                    name=f"run_facts_{session_id}",
                    tags=[session_id, "run_facts"],
                    when_to_use=f"Facts established earlier in session {session_id}.",
                )
                if entity_id:
                    logger.info("MemoryManager: saved run facts as %s", entity_id)
                return entity_id
        except Exception as exc:
            logger.warning("MemoryManager: save_run_facts failed: %s", exc)
            return None

    async def recall_run_facts(self) -> str:
        """Return facts saved earlier in the current session as a digest prefix
        (T33), or ``""`` when there are none / memory is off. Cards from another
        session or past their ``expires_at`` are filtered out."""
        session_id = self._config.memory_session_id
        if not (self._config.enable_memory and self._available and session_id):
            return ""
        try:
            MemoryClient = _memory_sdk().MemoryClient  # noqa: N806 (SDK class)
            now = time.time()
            with MemoryClient(base_url=self._config.memory_base_url) as client:
                hits = self._search_cards(client, session_id, top_k=self._config.memory_recall_top_k)
            lines: list[str] = []
            for h in hits:
                content = getattr(h, "content", None) or {}
                if content.get("kind") != "run_facts" or content.get("session_id") != session_id:
                    continue
                exp = content.get("expires_at")
                if isinstance(exp, int | float) and exp < now:
                    continue
                facts = content.get("facts")
                if not facts:
                    continue
                q = str(content.get("question", "")).strip()
                rendered = json.dumps(facts, ensure_ascii=False)
                lines.append(f"- {q} → {rendered}" if q else f"- {rendered}")
            if not lines:
                return ""
            return (
                "Facts already established earlier in this session "
                "(reuse instead of re-deriving):\n" + "\n".join(lines[:8])
            )
        except Exception as exc:
            logger.warning("MemoryManager: recall_run_facts failed: %s", exc)
            return ""

    async def save_agent_skill(
        self,
        resolved_skill: dict[str, Any],
        *,
        entity_id: str | None = None,
    ) -> str | None:
        """Persist AgentSkill provenance to memory (§5 P1).

        Delegates to ``gigaevo_client.MemoryClient.ingest_skill_from_carl``
        when the SDK is installed; falls back to the legacy
        ``gigaevo_memory.MemoryClient.save_chain`` shape when only the
        older client is available. Lazy-imported and exception-isolated
        so absence of either dep is non-fatal.

        Parameters
        ----------
        resolved_skill : dict[str, Any]
            Skill descriptor. Required keys: ``name``, ``uri``. Optional
            keys: ``description``, ``sha256``, ``manifest_summary``,
            ``tags``, ``allowed_tools``, ``relevance``.
        entity_id : str | None
            Pre-computed entity id for idempotency. When supplied, the
            SDK helper no-ops if the SHA already matches the stored
            entity. When omitted, the SDK derives an id from the SHA.

        Returns
        -------
        str | None
            The persisted ``entity_id``, or ``None`` if memory is
            disabled / both SDKs are missing / the call failed.
        """
        if not self._config.enable_memory:
            logger.debug("MemoryManager: memory disabled, skipping save_agent_skill")
            return None

        # Prefer the typed helper from gigaevo-client (§1.3).
        client_v2_cls: Any
        try:
            from gigaevo_client import MemoryClient  # noqa: N813

            client_v2_cls = MemoryClient
        except ImportError:
            client_v2_cls = None

        if client_v2_cls is not None:
            try:
                with client_v2_cls(base_url=self._config.memory_base_url) as client:
                    if not hasattr(client, "ingest_skill_from_carl"):
                        logger.debug(
                            "MemoryManager: gigaevo_client.MemoryClient lacks "
                            "ingest_skill_from_carl; falling back to legacy save"
                        )
                    else:
                        ref = client.ingest_skill_from_carl(resolved_skill, entity_id=entity_id)
                        ref_id = getattr(ref, "entity_id", None) or (
                            ref.get("entity_id") if isinstance(ref, dict) else None
                        )
                        if ref_id:
                            logger.info("MemoryManager: ingested agent_skill as %s", ref_id)
                            return str(ref_id)
            except Exception as exc:
                logger.warning("MemoryManager: ingest_skill_from_carl failed (%s); falling back", exc)

        # Legacy fallback via gigaevo-memory.MemoryClient.save_chain.
        if not self._available:
            logger.warning("MemoryManager: gigaevo-memory not installed, skipping save_agent_skill")
            return None

        try:
            MemoryClient = _memory_sdk().MemoryClient  # noqa: N806 (SDK class)

            name = str(resolved_skill.get("name") or "unnamed-skill")
            tags = ["mage-agent-skill"] + list(resolved_skill.get("tags") or [])
            with MemoryClient(base_url=self._config.memory_base_url) as client:
                ref = client.save_chain(
                    chain={"kind": "agent_skill", **resolved_skill},
                    name=f"mage_skill_{name}",
                    tags=tags,
                )
                eid: str = ref.entity_id
                logger.info("MemoryManager: saved agent_skill (legacy path) as %s", eid)
                return eid
        except Exception as exc:
            logger.warning("MemoryManager: save_agent_skill failed: %s", exc)
            return None

    async def save_reusable_step(
        self,
        step: dict[str, Any],
        *,
        domain: str,
        extra_tags: list[str] | None = None,
    ) -> str | None:
        """Persist an individual reusable LLM step as a ``step`` entity (§5 P1).

        Caller is expected to gate on :func:`is_reusable_step` to avoid
        cluttering memory with one-shot steps that won't generalise.
        The saved entity is tagged ``template`` so MAGE's
        :class:`TemplateAssemblerAgent` can recall it on subsequent runs.

        Parameters
        ----------
        step : dict[str, Any]
            CARL step dict (after `to_carl_compat`).
        domain : str
            Domain label for tagging.
        extra_tags : list[str] | None
            Extra tags to attach (e.g. detected capability tags).

        Returns
        -------
        str | None
            The persisted ``entity_id``, or ``None`` on failure.
        """
        if not self._config.enable_memory or not self._available:
            return None

        try:
            MemoryClient = _memory_sdk().MemoryClient  # noqa: N806 (SDK class)

            number = step.get("number") or 0
            title = str(step.get("title") or f"step-{number}")
            tags = [domain, "mage-step", "template"] + list(extra_tags or [])

            with MemoryClient(base_url=self._config.memory_base_url) as client:
                ref = client.save_chain(
                    chain={"kind": "reusable_step", "domain": domain, "step": step},
                    name=f"mage_step_{domain}_{title}",
                    tags=tags,
                )
                entity_id: str = ref.entity_id
                logger.info("MemoryManager: saved reusable step as %s", entity_id)
                return entity_id
        except Exception as exc:
            logger.warning("MemoryManager: save_reusable_step failed: %s", exc)
            return None

    async def save_intermediate_artifacts(
        self,
        artifacts: dict[str, Any],
        *,
        domain: str,
        query: str | None = None,
        chain_memory_key: str | None = None,
    ) -> list[str]:
        """Persist deep-mode intermediate artifacts as audit cards (§4 P2).

        Saves one ``memory_card`` entity per stage entry in
        ``artifacts``, tagged ``mage_intermediate`` + ``{domain}`` +
        ``mage_intermediate_{stage}`` so future runs (and CARE's
        audit UI) can recall the per-stage output without re-running
        generation. Best-effort: failures are logged and skipped, the
        caller's generation result is never affected.

        Parameters
        ----------
        artifacts : dict[str, Any]
            Stage name → artifact dict, matching the layout of
            :attr:`MAGEResult.intermediate_artifacts`. Typical keys:
            ``domain_analysis``, ``step_plan``, ``dag``,
            ``described_steps``, ``critique``, ``verification``,
            ``refine``.
        domain : str
            Domain label for tagging.
        query : str | None
            Original user query, persisted alongside each card so the
            audit UI can group cards by source query.
        chain_memory_key : str | None
            Memory key of the parent chain (when known) — stored on
            each card so the audit UI can link cards back to their
            chain entity.

        Returns
        -------
        list[str]
            Entity IDs persisted, in stage-iteration order. Empty
            when memory is disabled, unavailable, or every save
            failed.
        """
        if not self._config.enable_memory or not self._available:
            return []
        if not artifacts:
            return []

        saved: list[str] = []
        try:
            MemoryClient = _memory_sdk().MemoryClient  # noqa: N806 (SDK class)

            with MemoryClient(base_url=self._config.memory_base_url) as client:
                for stage_name, artifact in artifacts.items():
                    try:
                        payload: dict[str, Any] = {
                            "kind": "intermediate_artifact",
                            "stage": stage_name,
                            "domain": domain,
                            "artifact": artifact,
                        }
                        if query is not None:
                            payload["query"] = query
                        if chain_memory_key is not None:
                            payload["chain_memory_key"] = chain_memory_key

                        ref = client.save_chain(
                            chain=payload,
                            name=f"mage_intermediate_{domain}_{stage_name}",
                            tags=[domain, "mage_intermediate", f"mage_intermediate_{stage_name}"],
                        )
                        entity_id: str = getattr(ref, "entity_id", "")
                        if entity_id:
                            saved.append(entity_id)
                            logger.info(
                                "MemoryManager: saved intermediate artifact for %s as %s",
                                stage_name,
                                entity_id,
                            )
                    except Exception as exc:  # noqa: BLE001 — per-stage best-effort
                        logger.warning(
                            "MemoryManager: save_intermediate_artifacts failed for stage %s: %s",
                            stage_name,
                            exc,
                        )
        except Exception as exc:
            logger.warning("MemoryManager: save_intermediate_artifacts (top-level) failed: %s", exc)

        return saved

    async def save_failure_card(
        self,
        *,
        query: str,
        domain: str,
        reason: str,
        stage: str,
        chain_dict: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str | None:
        """Save a ``memory_card`` describing a failed generation (§5 P2).

        Persisted with tags ``mage-failure`` + ``{domain}`` so
        :class:`FeedbackRecallAgent` (already filtered to surface
        feedback for the current domain) recalls it on similar future
        queries. Best-effort: never raises.

        Parameters
        ----------
        query : str
            Original user query that produced the failed chain.
        domain : str
            Domain label (e.g. ``"finance"``).
        reason : str
            One-sentence description of what went wrong.
        stage : str
            Pipeline stage where the failure was observed
            (``"chain_verification"`` | ``"self_refine"`` | ...).
        chain_dict : dict[str, Any] | None
            The failed chain dict, embedded for postmortem. Optional —
            kept out of the payload when ``None`` to keep the card small.
        extra : dict[str, Any] | None
            Stage-specific signals (verification questions answered
            ``False``, refine quality before/after, etc.).

        Returns
        -------
        str | None
            The persisted ``entity_id``, or ``None`` on failure / when
            memory is disabled / unavailable.
        """
        if not self._config.enable_memory or not self._available:
            return None

        try:
            MemoryClient = _memory_sdk().MemoryClient  # noqa: N806 (SDK class)

            payload: dict[str, Any] = {
                "kind": "failure_card",
                "query": query,
                "domain": domain,
                "reason": reason,
                "stage": stage,
            }
            if chain_dict is not None:
                payload["chain"] = chain_dict
            if extra:
                payload["extra"] = extra

            with MemoryClient(base_url=self._config.memory_base_url) as client:
                ref = client.save_chain(
                    chain=payload,
                    name=f"mage_failure_{domain}_{stage}",
                    tags=[domain, "mage-failure", f"mage-failure-{stage}"],
                )
                entity_id: str = ref.entity_id
                logger.info("MemoryManager: saved failure card as %s (stage=%s)", entity_id, stage)
                return entity_id
        except Exception as exc:
            logger.warning("MemoryManager: save_failure_card failed: %s", exc)
            return None

    async def save_failure_cards_for_run(
        self,
        query: str,
        *,
        domain: str = "general",
        chain_dict: dict[str, Any] | None = None,
        step_results: list[Any] | None = None,
    ) -> list[str]:
        """Auto-write a failure card for each masked / explicit tool error in a
        run's executed steps (T34), so the evolve / feedback loop learns from
        EXECUTION failures — not only generation-stage rejections. CARE passes the
        replay's step results after a run. No-op unless ``enable_failure_card_save``
        is set (and memory is available). Returns the saved card keys.
        """
        if not self._config.enable_failure_card_save:
            return []
        keys: list[str] = []
        for f in find_masked_tool_errors(step_results):
            label = "masked tool error" if f.get("masked") else "tool error"
            key = await self.save_failure_card(
                query=query,
                domain=domain,
                reason=f"step {f.get('number')} ({f.get('tool_name') or 'tool'}): {label} — {f.get('error')}",
                stage="execution",
                chain_dict=chain_dict,
                extra={"tool_error": f},
            )
            if key:
                keys.append(key)
        return keys

    async def search_feedback(
        self,
        domain: str,
        task_type: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for chain_feedback entities in memory.

        Parameters
        ----------
        domain : str
            Knowledge domain to filter by.
        task_type : str
            Task type to filter by.
        top_k : int
            Maximum number of results.

        Returns
        -------
        list[dict[str, Any]]
            Matching feedback records, or empty list on failure.
        """
        if not self._config.enable_memory or not self._available:
            return []

        try:
            MemoryClient = _memory_sdk().MemoryClient  # noqa: N806 (SDK class)

            with MemoryClient(base_url=self._config.memory_base_url) as client:
                # P1.18 item 3: feedback lives as memory cards; search_hits keeps
                # the raw content so the flattened feedback fields survive.
                hits = self._search_cards(client, f"chain feedback {domain} {task_type}", top_k=top_k)
                return [
                    {
                        "chain_entity_id": c.get("chain_entity_id") or getattr(h, "entity_id", ""),
                        "domain": c.get("domain", domain),
                        "quality_score": c.get("quality_score", 0.0),
                        "success": c.get("success", True),
                        "failed_at_step": c.get("failed_at_step"),
                        "failure_reason": c.get("failure_reason"),
                        "pattern_used": c.get("pattern_used", ""),
                        "step_types_used": c.get("step_types_used", []),
                        "score": getattr(h, "score", 0.0),
                    }
                    for h in hits
                    if (c := getattr(h, "content", None) or {}).get("kind") == "mage_feedback"
                ]
        except Exception as exc:
            logger.warning("MemoryManager: search_feedback failed: %s", exc)
            return []

    async def save_ideas(
        self,
        ideas: list[dict[str, Any]],
        domain: str,
        query: str,
    ) -> str | None:
        """Save generated step ideas to memory.

        Parameters
        ----------
        ideas : list[dict[str, Any]]
            List of idea dictionaries to save.
        domain : str
            Domain label.
        query : str
            Original user query.

        Returns
        -------
        str | None
            The entity_id on success, or ``None`` on failure.
        """
        if not self._config.enable_memory or not self._available:
            return None

        try:
            MemoryClient = _memory_sdk().MemoryClient  # noqa: N806 (SDK class)

            with MemoryClient(base_url=self._config.memory_base_url) as client:
                # P1.18 item 3: ideas persist as a typed memory card.
                entity_id = self._save_card(
                    client,
                    {"kind": "mage_ideas", "ideas": ideas, "query": query},
                    name=f"mage_ideas_{domain}",
                    tags=[domain, "mage-idea"],
                    when_to_use=f"Generated step ideas for {domain} queries.",
                )
                if entity_id:
                    logger.info("MemoryManager: saved ideas as %s", entity_id)
                return entity_id
        except Exception as exc:
            logger.warning("MemoryManager: failed to save ideas: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Benchmarks (evolve mode)
    # ------------------------------------------------------------------

    async def save_benchmark(
        self,
        *,
        domain: str,
        tasks: list[dict[str, Any]],
        best_chain: dict[str, Any],
        best_score: float,
        query: str,
    ) -> str | None:
        """Save a synthesized benchmark + best evolved chain to memory.

        Parameters
        ----------
        domain : str
            Domain label, used for tag-based recall.
        tasks : list[dict[str, Any]]
            Serialized :class:`SynthesizedTask` instances.
        best_chain : dict[str, Any]
            Best evolved chain on this benchmark (CARL-compat dict).
        best_score : float
            Score of the best chain on the benchmark (0-1).
        query : str
            Original user query that produced this benchmark.

        Returns
        -------
        str | None
            The ``entity_id`` on success, or ``None`` on failure or when
            memory is disabled / unavailable.
        """
        if not self._config.enable_memory or not self._available:
            return None

        try:
            MemoryClient = _memory_sdk().MemoryClient  # noqa: N806 (SDK class)

            payload = {
                "kind": "mage_benchmark",
                "query": query,
                "domain": domain,
                "tasks": tasks,
                "best_chain": best_chain,
                "best_score": best_score,
                "task_count": len(tasks),
            }
            with MemoryClient(base_url=self._config.memory_base_url) as client:
                # P1.19: benchmarks persist as typed memory cards (not the `chain`
                # entity — they are not reasoning chains, and conflating them
                # pollutes chain search). Mirrors the P1.18 template/feedback migration.
                entity_id = self._save_card(
                    client,
                    payload,
                    name=f"mage_benchmark_{domain}",
                    tags=[domain, "mage-benchmark"],
                    when_to_use="A synthesized benchmark + best evolved chain for a domain.",
                )
                if entity_id:
                    logger.info("MemoryManager: saved benchmark as %s (%d tasks)", entity_id, len(tasks))
                return entity_id
        except Exception as exc:
            logger.warning("MemoryManager: save_benchmark failed: %s", exc)
            return None

    async def recall_benchmarks(
        self,
        *,
        domain: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Recall past synthesized benchmarks for the given domain.

        Returns the raw payloads saved by :meth:`save_benchmark` so the
        caller (BenchmarkSynthesizer) can pull the ``tasks`` field and
        fold them into the new run.

        Parameters
        ----------
        domain : str
            Domain to look up. Matched against the ``mage-benchmark`` tag
            namespace and the saved ``domain`` field.
        top_k : int
            Maximum number of benchmark entries to return.

        Returns
        -------
        list[dict[str, Any]]
            Each dict has at least ``tasks``, ``query``, ``best_score``,
            and ``entity_id``. Empty list on failure.
        """
        if not self._config.enable_memory or not self._available:
            return []

        try:
            MemoryClient = _memory_sdk().MemoryClient  # noqa: N806 (SDK class)

            with MemoryClient(base_url=self._config.memory_base_url) as client:
                # P1.19: recall from typed memory cards (search_hits exposes the raw
                # card content + relevance score), filtered by kind — matching how
                # save_benchmark now persists them.
                hits = self._search_cards(client, f"mage_benchmark {domain}", top_k=top_k)
                out: list[dict[str, Any]] = []
                for h in hits:
                    payload = getattr(h, "content", None) or {}
                    if not isinstance(payload, dict):
                        continue
                    if payload.get("kind") != "mage_benchmark":
                        continue
                    if domain and payload.get("domain") not in (domain, "general"):
                        continue
                    out.append(
                        {
                            "entity_id": getattr(h, "entity_id", ""),
                            "tasks": payload.get("tasks", []),
                            "query": payload.get("query", ""),
                            "best_score": payload.get("best_score", 0.0),
                            "domain": payload.get("domain", domain),
                            "score": getattr(h, "score", 0.0),
                        }
                    )
                return out
        except Exception as exc:
            logger.warning("MemoryManager: recall_benchmarks failed: %s", exc)
            return []
