"""
LLM prompt templates for MAGE pipeline stages.

As of §8 P2 the source of truth is a YAML file
(``mmar_mage/prompts_data/prompts.yaml``) — power users can customise
any prompt per-deployment by either editing that file, pointing the
``MAGE_PROMPTS_PATH`` env var at a custom YAML, or programmatically
loading via :func:`mmar_mage.prompt_loader.load_prompts`.

For backwards compatibility every prompt is still re-exposed as a
module-level constant (``STEP_PLANNER_PROMPT``, ``DAG_BUILDER_PROMPT``,
...). Existing code that does ``from mmar_mage.prompts import X``
keeps working unchanged.
"""

from __future__ import annotations

import logging

from .prompt_loader import load_prompts
from .step_types import STEP_TYPE_DESCRIPTIONS, step_types_for

logger = logging.getLogger(__name__)

# Load every prompt body at import time. Failures here are fatal —
# the bundled YAML must always be present; CARE-time overrides via
# `MAGE_PROMPTS_PATH` are handled inside `default_prompts_path()`.
_PROMPTS: dict[str, str] = load_prompts()


def _bind(name: str) -> str:
    """Pull a prompt from the loader, with a clear error on miss."""
    try:
        return _PROMPTS[name]
    except KeyError as exc:
        raise KeyError(
            f"Prompt {name!r} not found in YAML. Expected key under the "
            f"top-level mapping. Available keys: {sorted(_PROMPTS.keys())}"
        ) from exc


# ---------------------------------------------------------------------------
# §8 P2 — re-expose every prompt as a module-level constant so the
# legacy `from mmar_mage.prompts import X` pattern keeps working.
# These bindings are evaluated at module import time, so an
# `MAGE_PROMPTS_PATH` override that's set in the environment when the
# module first loads will populate the constants from the custom YAML.
# ---------------------------------------------------------------------------

BENCHMARK_SYNTHESIZER_PROMPT: str = _bind("BENCHMARK_SYNTHESIZER_PROMPT")
CHAIN_EDIT_PROMPT: str = _bind("CHAIN_EDIT_PROMPT")
CHAIN_FLATTEN_EXECUTION_PROMPT: str = _bind("CHAIN_FLATTEN_EXECUTION_PROMPT")
CHAIN_JUDGE_PROMPT: str = _bind("CHAIN_JUDGE_PROMPT")
CHAIN_JUDGE_REAL_OUTPUT_PROMPT: str = _bind("CHAIN_JUDGE_REAL_OUTPUT_PROMPT")
CHAIN_VERIFICATION_PROMPT: str = _bind("CHAIN_VERIFICATION_PROMPT")
DAG_BUILDER_PROMPT: str = _bind("DAG_BUILDER_PROMPT")
DOMAIN_ANALYZER_PROMPT: str = _bind("DOMAIN_ANALYZER_PROMPT")
FAST_MODE_SYSTEM_PROMPT: str = _bind("FAST_MODE_SYSTEM_PROMPT")
FEEDBACK_DIGEST_PROMPT: str = _bind("FEEDBACK_DIGEST_PROMPT")
LLM_REPAIR_PROMPT: str = _bind("LLM_REPAIR_PROMPT")
MCTS_CRITIC_PROMPT: str = _bind("MCTS_CRITIC_PROMPT")
MCTS_PROPOSE_STEP_PROMPT: str = _bind("MCTS_PROPOSE_STEP_PROMPT")
MCTS_SIMULATE_PROMPT: str = _bind("MCTS_SIMULATE_PROMPT")
MEMORY_PURIFIER_PROMPT: str = _bind("MEMORY_PURIFIER_PROMPT")
MEMORY_SUB_QUERY_GENERATOR_PROMPT: str = _bind("MEMORY_SUB_QUERY_GENERATOR_PROMPT")
PLAN_SCORER_PROMPT: str = _bind("PLAN_SCORER_PROMPT")
SELF_REFINE_CRITIQUE_PROMPT: str = _bind("SELF_REFINE_CRITIQUE_PROMPT")
SELF_REFINE_REVISE_PROMPT: str = _bind("SELF_REFINE_REVISE_PROMPT")
STEP_CRITIC_PROMPT: str = _bind("STEP_CRITIC_PROMPT")
STEP_DESCRIBER_PROMPT: str = _bind("STEP_DESCRIBER_PROMPT")
STEP_PLANNER_PROMPT: str = _bind("STEP_PLANNER_PROMPT")
SUGGESTED_NAMING_PROMPT: str = _bind("SUGGESTED_NAMING_PROMPT")
TOT_BRANCH_SCORER_PROMPT: str = _bind("TOT_BRANCH_SCORER_PROMPT")
VERIFICATION_FIXER_PROMPT: str = _bind("VERIFICATION_FIXER_PROMPT")
WEB_DIGEST_PROMPT: str = _bind("WEB_DIGEST_PROMPT")
WEB_SUB_QUERY_GENERATOR_PROMPT: str = _bind("WEB_SUB_QUERY_GENERATOR_PROMPT")


# ---------------------------------------------------------------------------
# Helpers preserved verbatim from pre-YAML prompts.py (referenced by
# `mmar_mage.agents.step_describer.StepDescriber` to build a step-type
# constraint suffix). Keeping these here means the agent's import path
# (`from ..prompts import ... get_step_types_constraint`) stays valid.
# ---------------------------------------------------------------------------


def get_step_types_constraint(allowed: list[str] | None) -> str:
    """Return a prompt constraint clause for allowed step types.

    Parameters
    ----------
    allowed : list[str] | None
        Whitelist of step types, or ``None`` for no restriction.

    Returns
    -------
    str
        Empty string if no restriction, otherwise an explicit constraint.
    """
    if not allowed:
        return ""
    # P-1: filter through the registry — drops typos + MAGE-internal
    # eval/aggregator, so the planner is constrained to real CARL types only.
    types_str = ", ".join(f'"{t}"' for t in step_types_for(allowed))
    return (
        f"\n\nSTRICT CONSTRAINT: You MUST ONLY use step_type from: [{types_str}]. No other step types are permitted.\n"
    )


# ---------------------------------------------------------------------------
# Merged from mage-emnlp-ablations (P1-P3 prompt additions; inline constants).
# CARE's YAML-backed prompts above are unchanged; these new ones are inline.
# ---------------------------------------------------------------------------
import json  # noqa: E402
from typing import Any  # noqa: E402

BENCHMARK_RUN_PROMPT = """\
You are evaluating whether a generated reasoning chain would solve a set of test \
cases. You are given a summary of the chain's steps and a list of test cases \
(each: an id, an input, and a rubric of properties the output must satisfy). For \
each test case, judge whether the chain — AS DESIGNED — would produce an output \
that satisfies the rubric.

Return a JSON object:
{
  "results": [
    {"task_id": "<id>", "passed": true | false, "score": <0.0-1.0>, "reason": "<short>"}
  ]
}

Rules:
- Judge the chain's DESIGN: does it have the steps / tools / structure to satisfy
  the rubric? Do not assume a flawless executor or facts the chain can't obtain.
- ``score`` is your confidence the rubric is met; ``passed`` = score >= 0.5.
- On failure, ``reason`` must name the concrete gap (the missing step / capability).
- Rate EVERY test case, using its exact ``task_id``.
- Return ONLY valid JSON.
"""

EXECUTION_REPAIR_PROMPT = """\
You are repairing a CARL reasoning chain that FAILED some test cases. You are \
given the user query, the current chain (JSON), and the specific test cases it \
failed (each: the input, the rubric it must satisfy, and WHY it failed).

Revise the chain so it would pass the failing cases WITHOUT breaking what \
already works. Return the COMPLETE revised chain as JSON, the same shape as the \
input chain.

Rules:
1. Make the SMALLEST change that addresses the specific failures — add or clarify
   a step, fix a dependency, sharpen an aim, or add a tool / validation step.
2. Preserve the overall structure (name, search_config, max_workers) and step
   numbering where possible; keep dependencies acyclic and self-consistent.
3. Every ``llm`` step must keep a non-empty, concrete ``aim``.
4. Do NOT over-engineer — only fix what the failures call for.
5. Return ONLY valid JSON (the complete revised chain).
"""

REFLECTIVE_MUTATION_PROMPT = """\
You are improving a CARL reasoning chain using REFLECTION (GEPA-style). You will \
see the user query, the current chain (JSON), its overall score, and its \
per-task failures on a synthesized benchmark.

First DIAGNOSE the single most important *systematic* weakness behind the \
failures (not a one-off symptom). Then propose ONE concrete structural edit that \
follows from that diagnosis — a textual "gradient" toward a better chain.

Return JSON:
{
  "diagnosis": "<1-2 sentences: the root weakness behind the failures>",
  "edit_type": "insert_step" | "remove_step" | "swap_steps" | "rewrite_step" | "change_step_type",
  "target_step_number": <int or null>,
  "new_step": <full step object, or null if not applicable>,
  "rationale": "<one sentence: HOW this edit follows from the diagnosis>"
}

Rules:
1. The edit MUST follow from the diagnosis — not a generic tweak.
2. One structural change only; keep it small.
3. If inserting/rewriting an LLM step, fill aim, reasoning_questions, \
   step_context_queries, stage_action — non-empty, concrete, domain-specific.
4. Preserve all dependencies that aren't directly touched.
5. Return ONLY valid JSON.
"""

TOPOLOGY_SELECTOR_PROMPT = """\
You choose the STRUCTURAL TOPOLOGY for a reasoning chain. Given a user query and \
its domain analysis, pick the 1-3 topologies whose shape best fits the task.

Available topologies (name | when_to_use | step_types | complexity):
{topology_catalog}

Worked examples (query → best topology — why this SHAPE fits):
- "Summarize this article in three bullets" → pipeline — one straight track; each step feeds the next.
- "Compare AWS, GCP, and Azure on cost, security, and support, then recommend one" → diamond — fan one input into independent sub-analyses, then synthesize.
- "Combine these three research notes into one brief" → funnel — several independent inputs collapse into one synthesizer.
- "Translate each of these 50 product descriptions to French" → map_reduce — the SAME operation over many independent items, then collect.
- "Review this large source file for bugs" / "Analyze these 12 modules and summarize the findings" → map_reduce — outline, then read+analyse each chunk/file independently (fan-out), then merge. Large or multiple @-referenced files almost always want this, not one giant llm step.
- "Solve this tricky math word problem reliably" → self_consistency — sample the same prompt N times and take the consensus to cut variance.
- "Is this contract clause enforceable? Argue both sides" → debate — a contested judgement sharpened by opposing arguments + a judge.
- "Write a publishable abstract; correctness matters more than speed" → critique_revise_loop — quality over latency: draft, critique, revise.
- "What year did X happen? Be certain it is accurate" → chain_of_verification — draft, then verify the factual claims before answering.
- "Route this support ticket to billing, technical, or account" → router — classify and send to exactly ONE handler.
- "Handle this request mixing a data lookup, a calculation, and a summary" → mixture_of_experts — heterogeneous sub-tasks routed to specialists, then merged.
- "Research this online and compute a figure, iterating until you have the answer" → react — reason, call a tool, observe, repeat.
- "Plan a product launch with marketing, engineering, and legal sub-plans, then combine" → plan_and_execute — decompose into sub-tasks, execute each, assemble.
- "Book the flight and charge my card" → human_in_the_loop — a consequential action needs approval first.

Rules:
1. Prefer the SIMPLEST topology that fits — default to `pipeline` for ordinary \
   single-track tasks. Only escalate to a more complex shape when the query's \
   STRUCTURE clearly calls for it (use the worked examples above as a guide).
2. Return the BEST choice first. Include alternatives only when genuinely viable.
3. Higher `complexity` topologies must earn their cost.
4. Each `rationale` must name the STRUCTURAL feature of the task that calls for the \
   shape — independent sub-analyses, per-item / per-file / per-chunk work, contested \
   judgement, factual verification, routing, consensus, a human gate, a tool-loop, or \
   decomposition — NOT just restate the query.

Return JSON:
{{
  "candidates": [
    {{"name": "<topology name from the catalog>", "rationale": "<one line: why it fits>", "confidence": <0..1>}}
  ]
}}
Return ONLY valid JSON. Every `name` MUST be one of the catalog names.
"""

TOOL_NEED_DETECTOR_PROMPT = """\
You decide whether solving a query REQUIRES a tool that does not exist yet.

Most tasks are solved by reasoning (`llm` steps) or by common existing tools. \
Only flag a NEW tool when the task needs a concrete, reusable operation that a \
language model cannot reliably do inline — e.g. parsing a specific binary/file \
format, a deterministic data transform, a precise numeric/scientific routine, or \
calling a niche API. Do NOT propose a tool for things an `llm` step does well \
(summarizing, classifying, drafting, reasoning).

Return JSON:
{
  "needed": <true|false>,
  "description": "<if needed: one line — what the tool does, its input and output>",
  "rationale": "<if needed: why generic llm/existing tools are insufficient>"
}
Default to "needed": false when in doubt. Return ONLY valid JSON.
"""

SKILL_SYNTHESIZER_PROMPT = """\
You author a reusable AgentSkill for a task that no existing skill covers. A skill \
is a SKILL.md document (instructions an agent follows) plus, optionally, one \
helper Python script.

Return JSON:
{
  "name": "<snake_case skill identifier>",
  "description": "<one line: what the skill does and when to use it>",
  "instructions": "<markdown body: concrete, step-by-step guidance the agent follows; headings, numbered steps, examples>",
  "allowed_tools": "<optional space-separated tool tokens, e.g. 'Read Write Bash(python:*)' — omit or empty if none>",
  "script": {"filename": "<name>.py", "source": "<python module source>"} | null
}

Rules:
- `instructions` is the heart of the skill — actionable, specific, not vague.
- Include a `script` ONLY when the task needs deterministic code; otherwise null.
- A script must NOT use `eval`/`exec`/`compile`/`__import__`, `subprocess`, \
  `os.system`, shell-outs, or network access (it is sandbox-scanned and dropped if it does).
- Return ONLY valid JSON.
"""

TOOL_IMPL_PROMPT = """\
You implement a single Python tool from its spec. Output ONLY a self-contained \
Python module — no prose, no markdown fences.

Hard rules (the code is sandbox-validated and REJECTED on any violation):
- Define a top-level function named EXACTLY `{func_name}` matching the signature.
- NO `eval`, `exec`, `compile`, `__import__`.
- NO `subprocess`, `os.system`, `os.popen`, `os.exec*`, no shelling out.
- NO file deletion (`os.remove`/`os.unlink`/`os.rmdir`), no `pickle`/`marshal`.
- NO network access unless the module is listed in the declared dependencies.
- Prefer the standard library; keep it deterministic and single-purpose.
- Include a docstring and basic input validation; raise clear exceptions on bad input.

{repair_feedback}Spec follows. Return ONLY the module source.
"""

TOOL_SPEC_PROMPT = """\
You specify a single reusable tool as a DESIGN ARTIFACT (no code — just the spec).

Given a tool need, produce a precise, minimal, single-purpose tool. Prefer the \
standard library; only list third-party `dependencies` that are truly required.

Return JSON:
{
  "name": "<snake_case identifier>",
  "signature": "<python-style, e.g. extract_tables(pdf_path: str) -> list[dict]>",
  "docstring": "<one paragraph: behavior, assumptions, edge cases>",
  "input_schema": {"<arg>": "<type/description>"},
  "output_schema": {"<field>": "<type/description>"},
  "implementation_hint": "<how it would be built — libraries, approach>",
  "dependencies": ["<pypi package>", "..."],
  "examples": [{"input": {"<arg>": "<value>"}, "expected_output": "<value>", "comparison": "exact|contains|type_only"}]
}
Provide 2-4 concrete `examples` (input kwargs -> expected output) that pin the tool's behaviour — they are run as unit tests against the implementation, so a tool that merely "doesn't crash" but returns garbage is rejected. Use "exact" for deterministic outputs, "contains" when the output must include a value, "type_only" when only the return type is stable.
Keep it single-purpose and deterministic. Return ONLY valid JSON.
"""

STEP_PLANNER_ANTI_PATTERNS_PROMPT = """\

Avoid these anti-patterns:
- Don't use `supervisor` when all sub-tasks share the same logic — use a single \
step, `conditional`, or `transform` instead. Routing is only for genuinely \
heterogeneous pipelines.
- Don't add `parallel_sampling` to a step that runs deterministic logic (a tool \
call, a lookup, a fixed transform) — sampling only helps non-deterministic \
reasoning, otherwise it just wastes tokens.
- Don't emit a 6+ step chain for a single-tool or single-question task — reach \
for the fewest steps that solve it (prefer ≤4); justify any longer chain.
- Don't use `debate` for non-contested, factual, or single-correct-answer tasks \
— it's for close-call judgement, not lookups.
- Don't promote every step to `checkpoint=true` — mark only 1-3 at major phase \
boundaries (after expensive retrieval/extraction), never the final step.
- Don't add ceremonial "analyze the request" / "format the output" steps a single \
capable step already covers.
- Don't pack multiple external calls into one `tool`/`mcp` step — one tool step = one \
`tool_name` call. A live composite ("get my IP, then its country, then the weather \
there") is THREE dependent tool steps, not one step that "gets IP, geo and weather". \
The words "then call" / "and then" in a tool step's rough_aim mean you must split it.
- Don't bake a remembered personal fact (country, city, IP — "the user is in Russia") \
into a `tool`/`mcp` step's aim when that step is exactly what DISCOVERS it live. Memory \
is stale or wrong (a VPN put the user in Sweden, not the remembered Russia): a fetch \
step must DETERMINE the value, and downstream steps consume it via $history — never \
assume the result inside the aim of the step that produces it.
- Match the fetch tool to the OUTPUT shape, and produce STRUCTURED fields with a \
`structured_output` step (NOT a `transform`). `web_search` returns PROSE — for a result \
with named fields (temperature, date, …), follow it with a `structured_output` step whose \
LLM reads the prose and fills the schema. Reach for `http_request` only when you can \
supply a COMPLETE url STRING (e.g. "https://api.open-meteo.com/v1/forecast?latitude=..&longitude=.."): \
a tool step CANNOT carry a separate params/headers dict — input_mapping values are \
strings — so bake the query into the URL (build it in a prior `transform` "format" step \
if it depends on earlier output) and map the tool's url field to that built string. Don't \
over-split: one listed tool that covers the task is one step; use N steps only for N real \
external calls.
- Don't add a `human_input` step to obtain the agent's PRIMARY input — that value is \
ALREADY `$outer_context` (the user's request, or a deployed call's "input"). Reference \
`$outer_context` directly in the first step that consumes it (e.g. a weather agent that \
"receives a city" maps the fetch step to `$outer_context`; it does NOT ASK for the city). \
`human_input` is only for a clarification the original request genuinely lacks, asked \
mid-run — and a DEPLOYED agent has no human to answer, so such a step stalls it.
"""

ADVANCED_STEP_EXAMPLES: list[dict[str, Any]] = [
    {
        "step_type": "evaluation",
        "number": 3,
        "title": "Validate extracted records",
        "aim": "Check the step-2 records satisfy the schema and completeness criteria",
        "dependencies": [2],
        "evaluation_target_step": 2,
        "evaluation_criteria": ["all required fields present", "no null primary keys"],
        "evaluation_method": "llm",
        "evaluation_fail_action": "retry_with_feedback",
    },
    {
        "step_type": "parallel_sampling",
        "number": 2,
        "title": "Solve via self-consistency",
        "aim": "Independently derive the answer several times and keep the majority",
        "dependencies": [1],
        "sampling_n": 5,
        "sampling_aggregation": "vote",
    },
    {
        "step_type": "agent_skill",
        "number": 1,
        "title": "Extract tables from the PDF",
        "aim": "Use the pdf skill to extract every table as structured rows",
        "dependencies": [],
        "skill": "github://anthropics/skills/document-skills/pdf@main",
        "skill_task": "extract all tables as CSV-ready rows",
        "skill_execution_mode": "subagent",
    },
    {
        "step_type": "supervisor",
        "number": 1,
        "title": "Route the request to a specialist",
        "aim": "Dispatch the request to the pipeline that best fits it",
        "dependencies": [],
        "supervisor_routes": {"billing": "chains/billing.json", "technical": "chains/tech.json"},
        "supervisor_routing_prompt": "Choose 'billing' for payment/invoice issues, else 'technical'.",
    },
    {
        "step_type": "debate",
        "number": 2,
        "title": "Debate the recommendation",
        "aim": "Have a proponent and a skeptic argue, then a judge synthesizes the verdict",
        "dependencies": [1],
        "debate_agents": [
            {"name": "proponent", "role": "Argue FOR the recommendation with evidence"},
            {"name": "skeptic", "role": "Surface risks and argue AGAINST"},
        ],
        "debate_rounds": 2,
        "debate_judge_prompt": "Weigh both sides and output the final recommendation with caveats.",
    },
]


def _render_advanced_examples(examples: list[dict[str, Any]]) -> str:
    body = "\n".join(json.dumps(ex, ensure_ascii=False) for ex in examples)
    return (
        "\n\n## Worked examples for advanced step types (REFERENCE ONLY — adapt the "
        "SHAPE to the actual plan; do NOT echo these verbatim):\n" + body + "\n"
    )


MEMORY_APPLICABILITY_PROMPT = """\
You are a strict relevance judge for a memory-retrieval system. Given the \
current user query and a list of recalled past items (each with an id, a name, \
and a short structure summary), rate how APPLICABLE each item is to solving the \
CURRENT query.

Return a JSON object:
{
  "ratings": [
    {"entity_id": "<id>", "applicability": <0.0-1.0>, "reason": "<one short phrase>"}
  ]
}

Scoring guidance:
- 1.0 = the past item's structure/approach directly transfers to this query.
- 0.5 = partially relevant; some steps or patterns reusable.
- 0.0 = unrelated; only superficial keyword overlap.
- Be STRICT: shared domain words are NOT applicability. When unsure, score low.
- Rate EVERY item you are given, using its exact entity_id.
- Return ONLY valid JSON.
"""

CAPABILITY_SYNTHESIS_PROMPT = """\
No capability registry was found for this environment. Based on the user's \
task, SUGGEST the external tools and MCP servers a reasoning chain would \
plausibly need — these are hints for the planner, not confirmed-available \
resources.

Return a JSON object:
{
  "tools": [
    {"name": "<snake_case_tool_name>", "description": "<what it does>",
     "when_to_use": "<which step would call it>"}
  ],
  "mcp_servers": [
    {"name": "<server_name>", "description": "<what it exposes>"}
  ]
}

Rules:
- Suggest only what the task genuinely implies (e.g. PDF task → a pdf_extract
  tool; data API task → an http_get / api_call tool). Prefer 1-4 tools.
- Pure-reasoning tasks need no tools → return empty lists.
- Use generic, recognizable names; do NOT invent vendor-specific products.
- Return ONLY valid JSON.
"""

FEEDBACK_APPLICABILITY_PROMPT = """\
You are a strict relevance judge for a planning system. Given the current user \
query and a list of PAST EXECUTION FEEDBACK records (each with an id, the DAG \
pattern used, the step types, whether it succeeded, and any failure reason), \
rate how APPLICABLE each record's lesson is to the CURRENT query.

Return a JSON object:
{
  "ratings": [
    {"id": "<id>", "applicability": <0.0-1.0>}
  ]
}

Scoring guidance:
- 1.0 = the past lesson (what worked or what failed) directly informs how to
  plan THIS query.
- 0.0 = unrelated; shared domain or task-type alone is NOT applicability.
- Be STRICT; when unsure, score low.
- Rate EVERY record, using its exact id.
- Return ONLY valid JSON.
"""

MEMORY_GAP_SUBQUERY_PROMPT = """\
A first memory-recall pass for the user's query returned too few applicable \
results. Your job is to generate a SECOND wave of sub-queries that target what \
the first pass MISSED — different angles, sub-problems, adjacent techniques, or \
more specific/more general phrasings — NOT rephrasings of what already matched.

Return a JSON object:
{
  "sub_queries": ["<gap-targeting query 1>", "<query 2>", ...]
}

Guidance:
- Look at the already-found items (provided below) and deliberately aim ELSEWHERE.
- Prefer concrete, retrieval-friendly phrasings over abstract ones.
- Generate at most the requested number of sub-queries.
- Return ONLY valid JSON.
"""

WEB_GAP_SUBQUERY_PROMPT = """\
A first web-research pass for the user's query produced a digest with too few \
well-sourced claims. Generate a SECOND wave of web search queries targeting the \
UNSUPPORTED or missing aspects — different angles, more specific phrasings, or \
authoritative-source-friendly terms — NOT rephrasings of what already worked.

Return a JSON object:
{
  "sub_queries": ["<gap-targeting query 1>", "<query 2>", ...]
}

Guidance:
- Look at the partial digest below and deliberately aim at its gaps.
- Prefer concrete, search-friendly phrasings.
- Generate at most the requested number of queries.
- Return ONLY valid JSON.
"""

PLAN_PAIRWISE_PROMPT = """\
You are a reasoning-plan judge. Given a user query, domain analysis, and TWO \
candidate step plans (A and B), decide which ONE is better for solving the task.

Return a JSON object:
{
  "winner": "A" | "B",
  "reason": "<one short sentence>"
}

Judge on (in order):
1. Completeness — covers all aspects of the query.
2. Right-sizing — neither over-engineered (redundant steps) nor under-specified.
3. Step ordering — logical dependencies; parallelism where it helps.
4. Specificity — concrete, actionable aims over vague ones.

Pairwise comparison is deliberate: judge the two plans head-to-head, not against
an absolute rubric. If they are equivalent, prefer the simpler (fewer steps).
Return ONLY valid JSON.
"""

VERIFICATION_ANSWER_PROMPT = """\
You are an INDEPENDENT verification answerer for reasoning-chain QA — you did \
NOT design this chain. Answer the verification question about the step under \
review honestly and skeptically, using ONLY the provided step and its \
dependency steps. For cross-step questions (dependencies, ordering, \
consistency) judge the step against the dependency steps you are given.

Return a JSON object:
{
  "answer": "yes" | "no",
  "explanation": "<brief, concrete>",
  "passed": true | false
}

Rules:
- ``passed=true`` only if the step's design is sound for the question's aspect.
- ``passed=false`` must name a concrete deficiency (the fixer will act on it).
- Do not assume facts not present in the provided steps.
- Return ONLY valid JSON.
"""

# NOTE: the worked examples for advanced step types (rendered from
# ``ADVANCED_STEP_EXAMPLES`` via ``_render_advanced_examples``) are baked
# directly into the ``STEP_DESCRIBER_PROMPT`` body in ``prompts_data/prompts.yaml``,
# so the module constant mirrors the YAML byte-for-byte (no post-bind mutation).
# ``ADVANCED_STEP_EXAMPLES`` remains the structured source used by
# ``tests/test_describer_examples.py`` to validate each example.


# P6.2 — needs-analysis: the model declares what the task REQUIRES so the
# pipeline can resolve (skills/tools/MCP) or request (user data / credentials)
# it before planning. Inline (not YAML-bound) like TOPOLOGY_SELECTOR_PROMPT.
NEEDS_ANALYSIS_PROMPT = """\
You analyse what a task NEEDS to be completed — the capabilities and inputs \
required — so the system can resolve or request them BEFORE planning.

Given the user query and its domain analysis, fill these lists:
- needs_skills: packaged skills required (e.g. pdf, docx, pptx, xlsx, webdev, mcp-builder)
- needs_tools: built-in tools (web_search, fetch_url, rest_call, run_python, read_file, file_outline)
- needs_services: external services to connect to (email/gmail, calendar, slack, a named API)
- needs_from_user: data or choices ONLY the user can provide (target language, which file, a date)
- needs_credentials: secrets/keys required (e.g. GMAIL_TOKEN, an API key) — names only, never values

Rules:
- List ONLY what is genuinely required. Leave a list empty when nothing applies.
- Do NOT invent needs the task doesn't have; do NOT include actual secret values.

Return JSON ONLY:
{{"needs_skills": [], "needs_tools": [], "needs_services": [], "needs_from_user": [], "needs_credentials": []}}
"""


# P6.3 — capabilities preamble: a precise "system card" so the planner KNOWS
# its full toolbox (every CARL step type, the bundled tools, attached-file I/O,
# and the human_input pause-and-ask mechanism) and stops defaulting to all-llm
# chains. Keys cover every type in ``schemas.CARL_VALID_STEP_TYPES`` (16).
# Single source of truth now lives in `step_types.py` (P-1); re-exported here so
# the legacy `from mmar_mage.prompts import STEP_TYPE_GUIDE` import keeps working.
STEP_TYPE_GUIDE: dict[str, str] = STEP_TYPE_DESCRIPTIONS


# P-2: per-type field guidance for the StepDescriber, keyed by step_type. The
# describer prompt is assembled from ONLY the sections for the types actually
# present in the DAG (saves ~1.5k tokens/call vs. listing all 16). MAGE-internal
# eval/aggregator are not here — the planner no longer emits them (P-1).
STEP_DESCRIBER_TYPE_SECTIONS: dict[str, str] = {
    "llm": 'step_type "llm": include aim, reasoning_questions, step_context_queries, stage_action, example_reasoning',
    "tool": 'step_type "tool": include tool_name, tool_input_mapping (reference a dependency by its step number, e.g. {"query": "$history[2]"} where 2 is listed in dependencies; "$history[-1]" is a last-resort fallback), tool_output_key, tool_timeout',
    "mcp": 'step_type "mcp": include mcp_server_name, mcp_tool_name, mcp_arguments or mcp_argument_mapping',
    "structured_output": 'step_type "structured_output": include structured_output_schema (a JSON Schema dict), structured_output_schema_name, structured_output_input_source (the dependency output by number, e.g. "$history[2]" where 2 is listed in dependencies; "$history[-1]" is a last-resort fallback), structured_output_instruction (free-form prompt to the model), structured_output_strict_json (bool, default true). Also fill aim/reasoning_questions/stage_action — the schema describes the SHAPE, the LLM still needs intent. This is the RIGHT way to turn a prior step\'s output (prose from web_search OR JSON from http_request) into an object with specific fields — the LLM extracts/derives each field. Use it (NOT a "transform format") for any "return X in structured form with fields A, B, C" task, and make it the FINAL step so its filled object is the chain\'s answer.',
    "conditional": 'step_type "conditional": include condition_branches ([{"condition": "expr", "next_step": N}]), condition_default_step',
    "memory": 'step_type "memory": include memory_operation ("read"|"write"|"append"|"delete"|"list"), memory_key, memory_value_source',
    "transform": 'step_type "transform": include transform_type ("extract"|"format"|"aggregate"|"filter"|"map"), transform_input_key, transform_expression. CRITICAL: transform is MECHANICAL string/data plumbing, NOT an LLM. "format" only substitutes WHOLE "$history[N]" / "$outer_context" values into a template — it does NOT slice ("$history[0][:10]" is NOT supported), does NOT call a model, and literal placeholders like "{{extracted from ...}}" are NEVER filled (they ship as-is and produce garbage). To EXTRACT or DERIVE fields from a prior step (e.g. pull temperature/condition out of a weather payload, parse a date, or build an object whose values are not literally present in the input), DO NOT use transform — use a "structured_output" step (an LLM reads the source and fills the schema).',
    "agent_skill": 'step_type "agent_skill": include skill (URI string — prefer "github://anthropics/skills/skills/<name>@main" for official skills, "local:///abs/path" for local, or just "<name>" for name lookup), skill_task (concrete task with {placeholder} refs), skill_execution_mode (one of: "llm" | "script" | "hybrid" | "subagent" | "llm_agent" — pick llm_agent for iterative tool-use, llm for single-shot, script for deterministic, subagent when raw data must be synthesised by LLM), skill_input_mapping (dict mapping {placeholder} keys to $history[<dependency step number>] / $memory.<ns>.<key>), skill_output_key (where to store result), skill_timeout (seconds, default 120).',
    "evaluation": 'step_type "evaluation": include evaluation_target_step (step number whose output to grade), evaluation_method (EXACTLY one of "rule" — simpleeval boolean expressions over `value`, or "llm" — an LLM judge over plain-English criteria; prefer "llm"), evaluation_criteria (a JSON LIST of strings — each one a pass condition, e.g. ["all required fields present", "no null primary keys"]; NEVER a bare string), evaluation_fail_action (EXACTLY one of "continue" | "abort" | "retry_with_feedback").',
    "parallel_sampling": 'step_type "parallel_sampling": include sampling_n (number of samples, e.g. 5), sampling_temperature (e.g. 0.7), sampling_aggregation ("vote" | "best_of" | "mean" | "concat"), sampling_judge_prompt (if aggregation=best_of, the judge prompt). Also fill aim/reasoning_questions for the body.',
    "agent_handoff": 'step_type "agent_handoff": include handoff_chain_path (file path or registry key for the sub-chain), handoff_input_mapping (dict), handoff_output_key.',
    "supervisor": 'step_type "supervisor": include supervisor_routes (dict mapping route name → sub-chain path), supervisor_routing_prompt (instructions for the routing LLM).',
    "debate": 'step_type "debate": include debate_agents (list of {name, role, llm_config}), debate_rounds (int), debate_judge_prompt.',
    "human_input": 'step_type "human_input": include human_prompt (what to ask), human_input_timeout, human_default (fallback).',
    "mcp_resource": 'step_type "mcp_resource": include mcp_resource_uri (URI of the resource), mcp_resource_memory_key (where to store).',
    "tool_discovery": 'step_type "tool_discovery": include discovery_sources (list of {kind: "module"|"callable"|"dict", ...}).',
}


def build_step_describer_prompt(dag_types: set[str] | list[str] | None = None) -> str:
    """Assemble ``STEP_DESCRIBER_PROMPT`` with only the type sections present in
    the DAG (P-2). When ``dag_types`` is ``None`` all registry types are
    included; unknown / MAGE-internal labels in ``dag_types`` are ignored.
    """
    types = step_types_for(list(dag_types) if dag_types else None)
    sections = "\n".join(STEP_DESCRIBER_TYPE_SECTIONS[t] for t in types if t in STEP_DESCRIBER_TYPE_SECTIONS)
    return STEP_DESCRIBER_PROMPT.replace("<<STEP_TYPE_SECTIONS>>", sections)


def build_dag_builder_prompt() -> str:
    """Assemble ``DAG_BUILDER_PROMPT`` with its pattern enum sourced from the
    topology library — the single source of pattern names (P-4). Lazy import
    avoids any prompts ↔ topology import cycle.
    """
    from .topology.library import describe_patterns_for_dag

    return DAG_BUILDER_PROMPT.replace("<<DAG_PATTERNS>>", describe_patterns_for_dag())


_MAX_TOOL_SIGNATURE_CHARS = 2400  # ~600 tokens — bounds the describer's tool block (T15)


def _cap_tool_lines(lines: list[str], max_chars: int) -> tuple[str, int]:
    """Keep whole tool-signature lines within ``max_chars`` (T15) so a large
    capability context can't bloat the describer prompt. Always keeps at least one
    line. Returns ``(joined_kept_lines, omitted_count)``."""
    if max_chars <= 0:
        return "\n".join(lines), 0
    kept: list[str] = []
    total = 0
    for line in lines:
        if kept and total + len(line) + 1 > max_chars:
            break
        kept.append(line)
        total += len(line) + 1
    return "\n".join(kept), len(lines) - len(kept)


def render_tool_signatures(
    tools: list[dict[str, Any]] | None, *, max_chars: int = _MAX_TOOL_SIGNATURE_CHARS
) -> str:
    """Render available tools' call signatures + input fields for the StepDescriber (T3).

    Threaded from the capability context so the describer fills the EXACT field
    names (e.g. ``http_request(url, method)``) instead of inventing / mis-mapping
    them — the failure that put ``$outer_context`` into a ``url`` field. Returns
    "" when no tools are known.
    """
    if not tools:
        return ""
    lines: list[str] = []
    for t in tools:
        name = str(t.get("name", "")).strip()
        if not name:
            continue
        sig = str(t.get("signature", "")).strip()
        if not sig:
            schema = t.get("input_schema")
            args = ", ".join(str(k) for k in schema) if isinstance(schema, dict) else ""
            sig = f"{name}({args})"
        desc = str(t.get("description", "")).strip()
        lines.append(f"- {sig}" + (f" — {desc[:120]}" if desc else ""))
    if not lines:
        return ""
    body, omitted = _cap_tool_lines(lines, max_chars)
    note = (
        f"\n- … ({omitted} more tool(s) available — name one in a step's tool_name if needed)"
        if omitted
        else ""
    )
    return (
        "\n\nAvailable tool signatures — for a `tool` step, fill these EXACT field names; "
        "every required field must be non-empty and type-appropriate (a url field needs a "
        "real URL or a $history[N] reference, never free text):\n" + body + note
    )

_DEFAULT_BUNDLED_TOOLS: tuple[str, ...] = (
    "web_search(query) — search the web for current information",
    "fetch_url(url) — download a page's readable text",
    "rest_call(url, method='GET', json_body=None, …) — call a JSON REST API",
    "run_python(code) — execute Python in an isolated sandbox",
    "read_file(path, offset=0, length=N) — read a byte slice of an @-attached file",
    "file_outline(path) — size + line count + top-level structure of an @-attached file",
    "translate(text, target_language) / wikipedia(query) / calculator(expr)",
)


def build_capabilities_preamble(
    *,
    skills_block: str = "",
    mcp_block: str = "",
    tools: tuple[str, ...] | None = None,
) -> str:
    """Render the capabilities system card (P6.3).

    Always lists every step type (:data:`STEP_TYPE_GUIDE`) + the bundled tools
    (incl. ``read_file`` / ``file_outline``) + the attached-file I/O and
    ``human_input`` mechanisms. ``skills_block`` (from
    ``skills.format_surfaced_skills``) and ``mcp_block`` are appended when given.
    """
    lines = [
        "## Your capabilities (use the RIGHT primitive — do NOT default to `llm`)",
        "",
        "STEP TYPES:",
    ]
    lines += [f"- {name}: {when}" for name, when in STEP_TYPE_GUIDE.items()]
    lines += ["", "BUNDLED TOOLS (call via a `tool` step):"]
    lines += [f"- {t}" for t in (tools or _DEFAULT_BUNDLED_TOOLS)]
    lines += [
        "",
        "ATTACHED FILES: read an @-referenced file with `read_file` / "
        "`file_outline` (large files in slices); write deliverables (pptx / "
        "xlsx / docx) to the artifacts dir.",
        "INTERACTION: when something is missing (a file choice, a value, a "
        "credential), add a `human_input` step to PAUSE and ASK the user — "
        "never guess.",
    ]
    if skills_block.strip():
        lines += ["", skills_block.strip()]
    if mcp_block.strip():
        lines += ["", "CONNECTED MCP SERVERS:", mcp_block.strip()]
    return "\n".join(lines)


def build_planning_guidance(allowed: list[str] | None = None) -> str:
    """P6.6 — a planner preamble that (1) makes the model THINK about the
    desired output before choosing steps, (2) lists the CARL step types it may
    use, and (3) states the hard rule that producing a FILE needs an
    ``agent_skill`` step (an ``llm`` step only writes text). Prepended to
    ``STEP_PLANNER_PROMPT`` so weaker models stop defaulting every step to
    ``llm``.

    P-1: when ``allowed`` is given the step-type list is filtered to it (e.g.
    ``allowed=["llm"]`` shows one type, not the full menu). Step-type lines come
    from the ``step_types`` registry (single source).
    """
    lines = [
        "## First, think about what the task actually needs",
        "Before planning steps, decide the user's desired OUTPUT and what it "
        "takes to produce it:",
        "- A text answer? An actual FILE (.pptx / .docx / .xlsx / .pdf)? "
        "Structured data? An outward action (send / post)?",
        "- Then pick the step types + skills that ACHIEVE it. Do NOT default "
        "every step to `llm` — choose the primitive that really does the job.",
        "",
        "## CARL step types you can use",
    ]
    lines += [f"- `{t}` — {STEP_TYPE_DESCRIPTIONS[t]}" for t in step_types_for(allowed)]
    lines += [
        "",
        "## Skills DO real work — not just text",
        "An `agent_skill` step runs a packaged skill that performs the task for "
        "real: `pptx` CREATES a .pptx file, `docx` a Word document, `xlsx` a "
        "spreadsheet, `pdf` a PDF. An `llm` step only writes TEXT — it CANNOT "
        "create a file.",
        "HARD RULE: if the user asks for a file / document / presentation / "
        "spreadsheet, the step that produces it MUST be an `agent_skill` with "
        "the matching skill — NEVER a plain `llm` step that merely 'describes' it.",
        "Example — «make a PPTX lecture»: plan `llm` steps to draft the content, "
        "then a FINAL `agent_skill` step (skill = `pptx`) that turns that content "
        "into the actual .pptx file — not an `llm` step that lists slides as text.",
        "",
    ]
    return "\n".join(lines)
