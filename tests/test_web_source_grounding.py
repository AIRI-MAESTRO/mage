"""Source-grounded web research: citations + second wave (plan item P1.2).

`WebResearchAgent` now presents numbered sources to the digest prompt (so claims
carry `[N]` citations), and runs a gap-targeting second wave when the digest
cites fewer than `web_min_sourced_claims` distinct sources. Tests use a
directly-mocked LLM and a mocked `_search_web` (no real network).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from mmar_mage.agents.web_research_agent import WebResearchAgent, _count_sourced_claims
from mmar_mage.schemas import DomainAnalysis, MAGEConfig

SAMPLE_DOMAIN = DomainAnalysis(domain="finance", task_type="analysis", complexity="medium")


def _agent(side_effect) -> WebResearchAgent:
    llm = MagicMock()
    llm.generate_json = AsyncMock(side_effect=side_effect)
    return WebResearchAgent(llm)


def _cfg(**kw) -> MAGEConfig:
    base = dict(
        api_key="t",
        enable_web_research=True,
        web_search_api_key="k",
        web_research_sub_queries=1,
        max_retries=0,
    )
    base.update(kw)
    return MAGEConfig(**base)


def _result(url: str) -> dict:
    return {"title": f"t-{url}", "url": url, "content": "snippet"}


# --------------------------------------------------------------------------- #
# _count_sourced_claims
# --------------------------------------------------------------------------- #
def test_count_distinct_markers():
    assert _count_sourced_claims("a [1] b [2] c [1] d [3]") == 3  # distinct


def test_count_zero_when_no_markers():
    assert _count_sourced_claims("no citations here") == 0
    assert _count_sourced_claims("") == 0


# --------------------------------------------------------------------------- #
# _create_digest presents NUMBERED sources to the prompt
# --------------------------------------------------------------------------- #
async def test_create_digest_numbers_sources():
    llm = MagicMock()
    llm.generate_json = AsyncMock(return_value={"digest": "ok"})
    agent = WebResearchAgent(llm)

    await agent._create_digest("q", [_result("http://a"), _result("http://b")])

    user_prompt = llm.generate_json.call_args.kwargs["user_prompt"]
    assert "[1]" in user_prompt and "[2]" in user_prompt
    assert "http://a" in user_prompt


# --------------------------------------------------------------------------- #
# Second wave
# --------------------------------------------------------------------------- #
async def test_second_wave_fires_when_undersourced(mocker):
    # wave-1 digest cites only 1 source (< 3) → second wave recalls + re-digests
    agent = _agent(
        side_effect=[
            {"sub_queries": ["q1"]},
            {"digest": "Only one claim [1]."},
            {"sub_queries": ["gap1"]},
            {"digest": "Now well sourced [1][2][3]."},
        ]
    )
    search = mocker.patch.object(
        agent,
        "_search_web",
        new_callable=AsyncMock,
        side_effect=[[_result("http://a")], [_result("http://b")]],
    )

    result = await agent.research("query", SAMPLE_DOMAIN, _cfg(web_min_sourced_claims=3))

    assert result.digest == "Now well sourced [1][2][3]."  # re-digested after wave 2
    assert "http://a" in result.sources and "http://b" in result.sources  # merged
    assert search.await_count == 2  # wave-1 + wave-2 searches


async def test_second_wave_skipped_when_well_sourced(mocker):
    agent = _agent(
        side_effect=[
            {"sub_queries": ["q1"]},
            {"digest": "Claims [1][2][3] are all sourced."},  # 3 distinct ≥ 3 → no second wave
        ]
    )
    search = mocker.patch.object(agent, "_search_web", new_callable=AsyncMock, return_value=[_result("http://a")])

    result = await agent.research("query", SAMPLE_DOMAIN, _cfg(web_min_sourced_claims=3))

    assert result.digest == "Claims [1][2][3] are all sourced."
    assert agent._llm.generate_json.await_count == 2  # sub-queries + digest only
    assert search.await_count == 1  # no second-wave search


async def test_second_wave_disabled_by_zero_threshold(mocker):
    agent = _agent(
        side_effect=[
            {"sub_queries": ["q1"]},
            {"digest": "Undersourced [1]."},  # would trigger a wave if enabled
        ]
    )
    mocker.patch.object(agent, "_search_web", new_callable=AsyncMock, return_value=[_result("http://a")])

    result = await agent.research("query", SAMPLE_DOMAIN, _cfg(web_min_sourced_claims=0))

    assert result.digest == "Undersourced [1]."
    assert agent._llm.generate_json.await_count == 2  # no gap-query call
