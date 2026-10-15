"""T35 — execute benchmark chains through the REAL mmar-carl runtime.

The flattened-prompt stub (``chain_executor._build_chain_prompt``) never
executed tool steps, so evaluation could not catch the 09.06 incident class
(``$outer_context`` urls, masked tool errors, missing retry/replan). This
module runs a chain exactly the way CARE / the agent hub do:
``ReasoningChain.from_dict(use_typed_steps=True)`` → ``execute_async`` on a
``ReasoningContext`` — with two injection points that keep benchmark runs
deterministic:

* ``api`` — any LLM client CARL accepts (``OpenAICompatibleClient`` live, a
  canned stub in tests);
* ``register_tools`` — receives the context before execution;
  :func:`register_benchmark_tools` ships a deterministic mock-HTTP tool set
  (fixed ip/geo/weather payloads, echo fallback) so tool steps run with **no
  network**.

Eval scores feed ``evolution_meta`` via :func:`to_evolution_meta` (the
standardised 6-field schema Memory's ``versions/beating`` reads) — versions
are immutable, so callers pass it to ``save_chain(evolution_meta=…)`` at
save time.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

ToolRegistrar = Callable[[Any], None]

#: Deterministic payloads for the benchmark tool set — the run-096 composite
#: (ip → geolocate → weather) executes end-to-end with no network.
CANNED_IP = "185.189.46.80"
CANNED_GEO = {"ip": CANNED_IP, "country": "Sweden", "city": "Stockholm"}
CANNED_WEATHER = {"location": "Stockholm", "temp_c": 4, "condition": "Light rain"}


async def execute_chain_via_carl(
    chain_dict: dict[str, Any],
    task_input: str,
    *,
    api: Any,
    register_tools: ToolRegistrar | None = None,
    timeout_s: float = 300.0,
) -> tuple[bool, str, int, str | None]:
    """Run one chain through the real CARL DAG executor.

    Returns ``(success, final_output, steps_completed, error)``.
    ``from_dict`` mutates its input, so the dict is deep-copied first.
    """
    from mmar_carl import Language, ReasoningChain, ReasoningContext

    try:
        chain = ReasoningChain.from_dict(
            copy.deepcopy(chain_dict), use_typed_steps=True
        )
        context = ReasoningContext(
            outer_context=task_input, api=api, language=Language.ENGLISH
        )
        if register_tools is not None:
            register_tools(context)
        async with asyncio.timeout(timeout_s):
            result = await chain.execute_async(context)
    except TimeoutError:
        return False, "", 0, f"chain run exceeded {timeout_s:.0f}s"
    except Exception as exc:  # noqa: BLE001 — surfaced as a failed execution
        return False, "", 0, f"{type(exc).__name__}: {exc}"
    steps_completed = len(getattr(result, "step_results", None) or [])
    error = getattr(result, "error_message", None)
    return bool(result.success), result.get_final_output(), steps_completed, error


def build_eval_llm_client(config: Any, model: str) -> Any:
    """OpenAI-compatible CARL client from the eval config (live runs)."""
    import os

    from mmar_carl import OpenAICompatibleClient
    from mmar_carl.llm import OpenAIClientConfig

    api_key = config.api_key or os.getenv("OPENROUTER_API_KEY") or "no-key"
    return OpenAICompatibleClient(
        OpenAIClientConfig(
            api_key=api_key,
            base_url=config.base_url,
            model=model,
            temperature=config.temperature,
        )
    )


def register_benchmark_tools(context: Any) -> None:
    """Deterministic mock-HTTP tool set for benchmark execution (no network).

    Mirrors the names generated chains actually reference (the agent
    template's read-only set): ``http_request``, ``fetch_url``,
    ``web_search``, ``calculator``, ``current_datetime``. HTTP-ish tools
    answer from canned payloads keyed on the url/query, so composite chains
    like run-096's ip→geo→weather execute end-to-end byte-identically.
    """

    async def http_request(url: str, method: str = "GET", **_: Any) -> str:
        return _canned_http(url)

    async def fetch_url(url: str) -> str:
        return _canned_http(url)

    async def web_search(query: str) -> str:
        return (
            f"[mock search for {query!r}] Stockholm weather: light rain, 4°C. "
            f"Source: mock://weather"
        )

    def calculator(expression: str) -> str:
        from simpleeval import simple_eval

        return str(simple_eval(expression))

    def current_datetime(timezone: str = "UTC") -> str:
        return "2026-06-10T12:00:00+00:00"  # frozen for determinism

    context.register_tool("http_request", http_request)
    context.register_tool("fetch_url", fetch_url)
    context.register_tool("web_search", web_search)
    context.register_tool("calculator", calculator)
    context.register_tool("current_datetime", current_datetime)


def _canned_http(url: str) -> str:
    lowered = (url or "").lower()
    if "ipify" in lowered or lowered.endswith("/ip"):
        return json.dumps({"ip": CANNED_IP})
    if "ip-api" in lowered or "geo" in lowered:
        return json.dumps(CANNED_GEO)
    if "wttr" in lowered or "weather" in lowered:
        return json.dumps(CANNED_WEATHER)
    return json.dumps({"url": url, "mock": True})


def to_evolution_meta(
    *,
    fitness_score: float,
    experiment_id: str | None = None,
    generation: int | None = None,
    objectives: dict[str, float] | None = None,
    parent_version_ids: list[str] | None = None,
    mutation_kind: str | None = None,
) -> dict[str, Any]:
    """Eval scores → the standardised evolution_meta schema.

    Memory's ``versions/beating`` (the C1 promotion gate's eval check) reads
    ``fitness_score`` from this shape. Versions are immutable — pass the
    result to ``save_chain(evolution_meta=…)`` when persisting the scored
    version; existing versions cannot be back-filled.
    """
    meta: dict[str, Any] = {"fitness_score": float(fitness_score)}
    if experiment_id is not None:
        meta["experiment_id"] = experiment_id
    if generation is not None:
        meta["generation"] = generation
    if objectives:
        meta["objectives"] = {k: float(v) for k, v in objectives.items()}
    if parent_version_ids:
        meta["parent_version_ids"] = list(parent_version_ids)
    if mutation_kind is not None:
        meta["mutation_kind"] = mutation_kind
    return meta
