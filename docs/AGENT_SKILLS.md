# Agent Skills in MAGE

How MAGE finds, ranks, and selects packaged
[AgentSkills](https://agentskills.io) for a chain — and how to make
your own skill discoverable.

## Sources

`CapabilityLookupAgent` (`mmar_mage/agents/capability_lookup_agent.py`)
is the single source of truth for the planner's capability surface. It
merges four streams into one deduplicated `agent_skills` list:

| Source | Gated by | Provides |
| --- | --- | --- |
| `capability_registry` memory entity | `enable_capability_lookup` | tools + MCP servers |
| `MemoryClient.find_capability_matches` | `enable_memory_skill_lookup` | promoted skills |
| `mmar_carl.skill_loader.SkillLoader.catalog_all` | `enable_local_skill_catalog` | locally-installed skills |
| `SkillDiscoveryAgent` (built-in registry + optional GitHub web fallback) | `enable_skill_discovery` (+ `enable_skill_web_discovery` for web) | recommended skills |

Sources merge in **memory → local → discovered** order. Dedup is by
SHA256 of `SKILL.md` (with `(uri, name)` fallback for entries that
don't carry a SHA, like web-discovered ones). When duplicates exist
the **earlier source wins** so the entry with richer metadata
(typically the memory-promoted one) survives.

## The single planner digest

After merging, `CapabilityContext.skills_digest()` renders the list
into a prompt-friendly block that the `STEP_PLANNER_PROMPT` consumes:

```
Recommended packaged skills (use as `agent_skill` steps with `skill` = URI):
- pdf (relevance=0.92, source=memory): task mentions PDFs
    URI: github://anthropics/skills/document-skills/pdf@main
    Description: Extract / produce PDF files
```

The §2 P1 planner-prompt rule turns this digest into a soft
requirement: when a sub-goal clearly matches a listed skill, the
planner emits an `agent_skill` step pointing at the URI instead of
re-implementing the work as a `llm` / `tool` step.

## Scoring (planner-side)

`_score_capability_match(rough_aim, capability)` blends two signals:

```
0.7 * jaccard(filtered_tokens(rough_aim), filtered_tokens(capability)) +
0.3 * capability.relevance
```

Token filtering drops stop-words and tokens shorter than 3 chars so
"the", "a", "data", "extract" don't dominate. A score of ≥ 0.5 is the
documented threshold for "prefer agent_skill over a custom LLM step"
(noted in the planner prompt; the LLM is the final arbiter — the
score isn't a hard gate).

The scorer is deterministic and dependency-free; the *real* semantic
match happens earlier in `SkillDiscoveryAgent`'s LLM rerank.

## Writing a discoverable skill description

The discovery LLM (`SKILL_RERANK_PROMPT`) and the planner LLM both
read the `description` and `capabilities` fields. To rank well:

1. **Lead with the verb + object pattern.** "Extract tables from PDF
   documents" ranks higher than "PDF helper".
2. **List concrete capabilities, not categories.** Use bullet phrasing
   in the `capabilities` list: `["extract text from PDF", "extract
   tables from PDF", "fill PDF forms"]`.
3. **Include domain words the user will likely use.** A skill named
   `mcp-builder` should mention "Model Context Protocol", "MCP
   server", "stdio transport" so queries mentioning any of those
   tokens find it.
4. **Tag with execution_modes_supported.** When a skill works as both
   `llm_agent` and `script`, list both — the planner picks the mode
   matching the surrounding chain (LLM-only chains will prefer
   `llm_agent`).
5. **Pin a SHA via `skill_sha256`.** Without it, dedup falls back to
   `(uri, name)` and the same skill installed locally AND promoted to
   memory will surface twice.

The full `KnownSkill` schema (registry side) is in
`mmar_mage/skills/registry.py`; the `AgentSkillEntry` schema (planner
side, after merging) is in
`mmar_mage/agents/capability_lookup_agent.py`.

## Registering a skill in Memory

Two paths:

### A) Promote a locally-installed skill

```python
from mmar_mage.memory import MemoryManager
from mmar_mage.schemas import MAGEConfig

mgr = MemoryManager(MAGEConfig(enable_memory=True))
resolved = {
    "name": "my-skill",
    "description": "...",
    "uri": "github://my-org/my-skill@main",
    "sha256": "<sha of SKILL.md>",
    "tags": ["documents"],
    "allowed_tools": ["read_file"],
    "manifest_summary": "Top-level helpers for X",
}
entity_id = await mgr.save_agent_skill(resolved, entity_id="my-skill")
```

When `gigaevo-client` is installed, this routes through the typed
`MemoryClient.ingest_skill_from_carl` helper (idempotent via SHA); a
duplicate SHA is a no-op. Without `gigaevo-client`, the legacy
`gigaevo-memory.save_chain` shape is used.

### B) Auto-save during generation

Set `MAGEConfig.enable_skill_provenance_save = True`. Every
`agent_skill` step in a generated chain triggers a save of the
resolved skill descriptor — letting future runs in the same domain
find it via Memory without re-running discovery.

## Caller-supplied capabilities (§2 P1)

When CARE has skills installed locally that aren't yet in Memory,
pass them directly:

```python
from mmar_mage import MAGEGenerator
from mmar_mage.agents.capability_lookup_agent import (
    AgentSkillEntry, CapabilityContext,
)

caps = CapabilityContext(
    agent_skills=[
        AgentSkillEntry(
            name="local-skill",
            uri="local:///opt/skills/x",
            description="...",
            sha256="...",
        ),
    ],
)
result = await gen.generate("query", capabilities=caps)
```

Caller-supplied `capabilities` **bypass** the internal
`CapabilityLookupAgent` entirely (precedence: caller > memory + local
merge). The generator's `deep_stages_completed` records
`"capability_lookup_injected"` instead of `"capability_lookup"` so
CARE telemetry can distinguish the two paths.

## See also

- `tests/test_capability_unified.py` — exercises the 4-source merge +
  dedup
- `tests/test_capability_p1.py` — scoring helper + caller-supplied
  injection
- `tests/test_integration_p2.py::test_capability_aware_planning_*` —
  end-to-end PDF-skill selection
