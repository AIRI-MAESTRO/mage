# MAGE ↔ CARE Integration Contract

The runtime contract between MAGE (chain generator) and CARE
(authoring UI / TUI). Every behaviour described here is exercised
end-to-end by the `tests/test_integration_p2.py` suite and the
dedicated per-feature tests linked at the bottom.

## Public entrypoint

```python
result: MAGEResult = await MAGEGenerator(config).generate(
    query: str,
    *,
    context_files: list[dict] | None = None,    # §7.3 — replay-from-library
    cancel: asyncio.Event | None = None,        # §3 P1 — Esc-key abort
    capabilities: CapabilityContext | None = None,  # §2 P1 — caller injection
)
```

Plus seven per-stage methods (§4 P1) for "regenerate stage X" UX:
`analyze_domain`, `plan_steps`, `build_dag`, `describe_steps`,
`critique_steps`, `verify_chain`, `refine`. They're stateless wrappers
around the same agents `generate()` uses — no memory save, no
`MAGEResult` wrapping — so CARE can drive them however it wants.

`MAGEResult.replay_from(stage_name, *, partial_state=None, query=None)`
re-runs the pipeline from one of `domain_analysis | step_planning |
dag_building | step_describing` reusing cached
`intermediate_artifacts`. `partial_state` lets CARE edit the artifact
at the replay point (e.g. `{"step_plan": edited_plan_dict}`).

## Progress callback

```python
class MAGEProgressCallback(Protocol):
    # Legacy, required:
    def on_stage_start(self, stage: str) -> None: ...
    def on_stage_complete(self, stage: str, result: Any) -> None: ...
    def on_error(self, stage: str, error: Exception) -> None: ...

    # Optional (probed via getattr — legacy callbacks keep working):
    def on_llm_chunk(self, stage: str, delta: str) -> None: ...
    def on_stage_progress(self, stage: str, artifact: Any) -> None: ...
    def on_retry(self, stage: str, attempt: int, error: Exception) -> None: ...
```

Stage names that fire `on_stage_start` / `_complete`: `domain_analysis`,
`memory_research`, `web_research`, `capability_lookup`,
`template_assembly`, `feedback_recall`, `step_planning`,
`tot_planning`, `dag_building`, `mcts_construction`, `step_describing`,
`step_critique`, `chain_verification`, `self_refine`,
`fast_generation`. CARE doesn't need to handle every one — start with
the legacy three and add the optional hooks as the TUI grows.

### Streaming (§3 P0)

`on_llm_chunk(stage, delta)` fires per chunk during streamable LLM
calls when `MAGEConfig.enable_streaming=True`. Currently wired for the
`step_describing` stage; other stages fall back to a single
non-streaming call.

### Cancellation (§3 P1)

```python
cancel = asyncio.Event()
asyncio.create_task(maybe_press_esc(cancel))
try:
    result = await gen.generate(query, cancel=cancel)
except MAGECancelledError as exc:
    care_tui.show(f"Cancelled at {exc.stage}")
```

Cancel is checked at every stage boundary plus inside the fast-mode
retry loop. The in-flight LLM call completes (its result is
discarded), then the next stage's pre-check raises. The exception
carries `stage: str` identifying where the abort happened so CARE can
show "Cancelled at step_planning".

### Retry visibility (§3 P1)

`on_retry(stage, attempt, error)` fires whenever `LLMClient` retries
after a transient failure AND a further attempt remains. The final
terminal failure does NOT fire `on_retry`; it propagates as a
`MAGELLMError` instead. Useful for "Retrying (2/3)…" TUI banners.

## Capability context (§2)

`CapabilityContext` is the planner's complete view of available
tools, MCP servers, and AgentSkills:

```python
CapabilityContext(
    tools=[{"name": "calculator"}, ...],
    mcp_servers=[{"name": "postgres", ...}, ...],
    agent_skills=[
        AgentSkillEntry(
            name="pdf",
            description="...",
            uri="github://anthropics/skills/pdf",
            manifest_summary="...",
            sha256="...",
            tags=["documents"],
            allowed_tools=["read_file"],
            source="memory" | "local" | "discovered",
            relevance=0.92,
            why="task mentions PDFs",
        ),
    ],
    environment_id="default",
)
```

CARE has two ways to populate it:

1. **Let MAGE discover** — set `enable_capability_lookup=True` and
   any combination of `enable_memory_skill_lookup`,
   `enable_local_skill_catalog`, `enable_skill_discovery`. The
   internal `CapabilityLookupAgent` merges sources and dedupes.

2. **Inject directly** — pass `capabilities=ctx` to `generate()`.
   This **bypasses** the internal lookup completely. Use when CARE
   has skills installed locally that aren't yet in Memory.

See `docs/AGENT_SKILLS.md` for skill ranking + scoring details.

## Output contract: `MAGEResult`

```python
class MAGEResult(BaseModel):
    chain_json: str                          # serialised CARL chain
    chain_dict: dict[str, Any]               # same, as a dict
    code: str | None                         # ChainBuilder fluent-API Python (optional)
    memory_key: str | None                   # parent chain entity id
    mode: Literal["fast", "deep"]
    metadata: MAGEMetadata                   # see §7.2
    intermediate_artifacts: dict[str, Any]   # §7.1 — per-stage dumps
    memory_keys: dict[str, Any]              # §7.1 — all entity ids
```

### `MAGEMetadata` — library-ready naming (§7.2)

CARE's `SaveAgentModal` pre-fills from these three fields:

- `suggested_display_name: str` — ≤60-char punchy noun phrase
- `suggested_description: str` — 1-2 sentence "does X, takes Y,
  returns Z"
- `suggested_tags: list[str]` — domain + task type + capability
  flags (`uses-pdf`, `uses-web`, `uses-memory`)

Deep-mode runs replace the heuristic defaults with a dedicated
LLM call when `enable_suggested_naming_llm=True` (default). All
three fields are always populated — never `None`.

### `intermediate_artifacts` keys

After a deep run with default options, the dict contains:
`domain_analysis`, `step_plan`, `dag`, `described_steps`. Optional
stages add `critique`, `revised_steps`, `verification`, `refine`.
Each value is a Pydantic `.model_dump()` of the corresponding
agent's output so the bundle round-trips through JSON.

### `memory_keys` shape

```python
{
    "chain": "<chain entity id>",                # always present when memory enabled
    "skills": ["<id>", ...],                     # §5 P1 — agent_skill provenance
    "reusable_steps": ["<id>", ...],             # §5 P1 — template-pattern saves
    "failure_cards": ["<id>", ...],              # §5 P2 — verifier / refine failures
    "intermediate_artifacts": ["<id>", ...],     # §4 P2 — audit cards
}
```

Lists are populated only when the corresponding `enable_*_save` flag
is on. Missing keys mean the save was off or failed.

### Replay bundle (§7.1)

`MAGEResult.to_care_dict()` → JSON-serializable bundle CARE persists
as a single `agent_run` artifact:

```python
{
    "schema_version": 1,
    "mode": "deep" | "fast",
    "chain_json": "...",
    "chain_dict": {...},
    "code": "..." | None,
    "memory_key": "..." | None,
    "metadata": {...},                  # MAGEMetadata.model_dump()
    "intermediate_artifacts": {...},
    "memory_keys": {...},
}
```

`MAGEResult.from_care_dict(data)` round-trips it back. Unknown
`schema_version` values log a warning but still load best-effort
(forward-compat).

## Memory artifact: `CareChainMetadata` block (§7.3)

When `enable_memory=True`, the chain entity's `content.metadata` slot
carries everything CARE needs to re-prime a `ReasoningContext` on
"Re-run from library":

```python
chain.content.metadata = {
    "task_description": str,        # the original query
    "context_files": [{"path", "sha256", "size_bytes"}, ...],
    "generated_by": "mage",
    "mage_metadata": {...},         # MAGEMetadata.model_dump()
    "display_name": str,
    "description": str,
    "evolution_meta": {...},        # §5 P2 — fitness priors (when enabled)
}
```

This is what `ReasoningContext.from_chain_inputs()` reads on the CARL
side. When `gigaevo-client` is installed, MAGE routes through
`CareChainMetadata.merge_into_content()`; otherwise a direct
dict-merge produces an equivalent shape.

## Profile presets (§8 P2)

Three TOML configs under `configs/` matching CARE's most common
launch profiles:

- `care_default.toml` — production OpenAI, streaming + naming on,
  memory + research + ecosystem-saves off (opt-in via settings)
- `care_local.toml` — local provider with `json_mode_fallback=true`
  and `max_retries=3`
- `care_research.toml` — full research stack + every quality stage +
  every ecosystem-save flag

Load with `MAGEConfig.from_toml("configs/care_default.toml")`.

## Error taxonomy

- `MAGEValidationError` — the generated chain failed structural /
  CARL validation. Includes `raw_output` for postmortem.
- `MAGELLMError` — every retry exhausted. Includes `stage` attribute.
- `MAGEMemoryError` — gigaevo-memory interaction failed. Never raised
  from the public `generate()` path — memory failures are non-fatal
  there; the error type is for low-level callers.
- `MAGECancelledError` — `cancel.set()` observed mid-pipeline.
  Includes `stage` attribute identifying the abort point.

## Stage-completed labels

`MAGEMetadata.deep_stages_completed: list[str]` records which
sub-stages ran. CARE's debug view shows this verbatim. Notable
distinguishing labels:

- `capability_lookup` — the unified §2 P0 stage ran the internal lookup
- `capability_lookup_injected` — caller passed `capabilities=` and the
  internal lookup was bypassed
- `domain_analysis_replay` / `step_planning_replay` / etc. — produced
  by `MAGEResult.replay_from`

## See also

- `docs/AGENT_SKILLS.md` — capability/skill discovery details
- `docs/ARCHITECTURE.md` — pipeline diagram + design principles
- `tests/test_integration_p2.py` — end-to-end contract assertions
- `configs/care_*.toml` — recommended config presets
