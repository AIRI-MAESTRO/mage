"""Tests for Pydantic schemas and CARL validation."""

from __future__ import annotations

from mmar_mage.schemas import (
    CARLChainSchema,
    CARLStepSchema,
    DAGStep,
    DAGStructure,
    DomainAnalysis,
    MAGEConfig,
    MAGEMetadata,
    MAGEResult,
    MemoryHit,
    MemoryResearchResult,
    PlannedStep,
    StepPlan,
    WebResearchResult,
)


class TestMAGEConfig:
    """Tests for MAGEConfig defaults and validation."""

    def test_defaults(self) -> None:
        cfg = MAGEConfig()
        assert cfg.mode == "deep"
        assert cfg.model == "gpt-4o"
        assert cfg.temperature == 0.2
        assert cfg.enable_memory is True
        assert cfg.default_max_workers == 2
        assert cfg.max_steps == 10
        assert cfg.min_steps == 2

    def test_custom_values(self) -> None:
        cfg = MAGEConfig(
            mode="fast",
            model="gpt-3.5-turbo",
            temperature=0.8,
            enable_memory=False,
            max_steps=15,
        )
        assert cfg.mode == "fast"
        assert cfg.model == "gpt-3.5-turbo"
        assert cfg.enable_memory is False

    def test_deep_toggles(self) -> None:
        cfg = MAGEConfig(
            enable_domain_analysis=False,
            enable_step_planning=True,
            enable_dag_optimization=False,
            enable_step_descriptions=True,
        )
        assert cfg.enable_domain_analysis is False
        assert cfg.enable_step_planning is True

    def test_new_fields_defaults(self) -> None:
        """New deep-research fields default to safe non-breaking values."""
        cfg = MAGEConfig()
        assert cfg.provider == "openai"
        assert cfg.ssl_verify is True
        assert cfg.http_timeout == 60.0
        assert cfg.openrouter_site_url is None
        assert cfg.enable_memory_research is False
        assert cfg.enable_web_research is False
        assert cfg.memory_recall_top_k == 5
        assert cfg.memory_relevance_threshold == 0.4
        assert cfg.memory_research_sub_queries == 3
        assert cfg.web_search_provider == "tavily"
        assert cfg.web_search_api_key is None
        assert cfg.web_research_max_results == 5
        assert cfg.web_research_sub_queries == 2
        assert cfg.cold_start_candidates == 1
        assert cfg.stage_llm_overrides == {}

    def test_v2_fields_defaults(self) -> None:
        """New v2 fields default to safe non-breaking values."""
        cfg = MAGEConfig()
        assert cfg.enable_capability_lookup is False
        assert cfg.capability_environment_id == "default"
        assert cfg.enable_template_assembly is False
        assert cfg.output_format == "json"

    def test_provider_values(self) -> None:
        for provider in ("openai", "openrouter", "local", "custom"):
            cfg = MAGEConfig(provider=provider)
            assert cfg.provider == provider

    def test_stage_llm_overrides(self) -> None:
        cfg = MAGEConfig(
            stage_llm_overrides={
                "domain_analysis": {"model": "gpt-4o-mini", "temperature": 0.0},
            }
        )
        # §6 P2: stage_llm_overrides is now a typed dict[str, StageLLMOverride].
        override = cfg.stage_llm_overrides["domain_analysis"]
        assert override.model == "gpt-4o-mini"
        assert override.temperature == 0.0

    def test_stage_llm_overrides_rejects_typos(self) -> None:
        """§6 P2: typos like 'temprature' must raise instead of being silently dropped."""
        import pytest as _pt

        with _pt.raises(ValueError):
            MAGEConfig(
                stage_llm_overrides={"domain_analysis": {"temprature": 0.0}},
            )


class TestMAGEResult:
    """Tests for the output contract."""

    def test_minimal_result(self) -> None:
        result = MAGEResult(chain_json='{"steps":[]}')
        assert result.mode == "deep"
        assert result.memory_key is None
        assert result.code is None
        assert result.metadata.domain == "general"

    def test_full_result(self) -> None:
        result = MAGEResult(
            chain_json='{"steps":[]}',
            chain_dict={"steps": []},
            memory_key="abc-123",
            mode="fast",
            metadata=MAGEMetadata(
                domain="finance",
                num_steps=3,
                generation_time_seconds=1.5,
                model="gpt-4o",
            ),
        )
        assert result.mode == "fast"
        assert result.memory_key == "abc-123"
        assert result.metadata.num_steps == 3


class TestMAGEMetadata:
    """Tests for new metadata fields."""

    def test_new_metadata_defaults(self) -> None:
        meta = MAGEMetadata()
        assert meta.memory_hits_used == 0
        assert meta.web_results_used == 0
        assert meta.was_cold_start is False

    def test_new_metadata_values(self) -> None:
        meta = MAGEMetadata(memory_hits_used=5, web_results_used=3, was_cold_start=True)
        assert meta.memory_hits_used == 5
        assert meta.web_results_used == 3
        assert meta.was_cold_start is True


class TestDomainAnalysis:
    """Tests for DomainAnalysis model."""

    def test_defaults(self) -> None:
        da = DomainAnalysis()
        assert da.domain == "general"
        assert da.complexity == "medium"
        assert da.suggested_step_count == 3
        assert da.preferred_step_types == []

    def test_from_dict(self) -> None:
        da = DomainAnalysis.model_validate(
            {
                "domain": "medicine",
                "task_type": "qa",
                "complexity": "complex",
                "key_concepts": ["diagnosis", "symptoms"],
                "suggested_step_count": 6,
            }
        )
        assert da.domain == "medicine"
        assert len(da.key_concepts) == 2

    def test_preferred_step_types(self) -> None:
        da = DomainAnalysis(
            domain="finance",
            preferred_step_types=["mcp", "transform", "llm", "eval"],
        )
        assert da.preferred_step_types == ["mcp", "transform", "llm", "eval"]


class TestStepPlan:
    """Tests for StepPlan and PlannedStep."""

    def test_step_plan(self) -> None:
        plan = StepPlan(
            steps=[
                PlannedStep(number=1, title="Step 1"),
                PlannedStep(number=2, title="Step 2", step_type="memory"),
            ],
            rationale="test",
        )
        assert len(plan.steps) == 2
        assert plan.steps[1].step_type == "memory"


class TestDAGStructure:
    """Tests for DAGStructure and DAGStep."""

    def test_dag_structure(self) -> None:
        dag = DAGStructure(
            steps=[
                DAGStep(number=1, title="A", dependencies=[]),
                DAGStep(number=2, title="B", dependencies=[]),
                DAGStep(number=3, title="C", dependencies=[1, 2]),
            ],
            max_workers=2,
            pattern="funnel",
        )
        assert dag.pattern == "funnel"
        assert dag.steps[2].dependencies == [1, 2]

    def test_new_dag_patterns(self) -> None:
        for pattern in ("critique_revise", "map_reduce", "multi_hop"):
            dag = DAGStructure(pattern=pattern)
            assert dag.pattern == pattern


class TestCARLSchemas:
    """Tests for CARLChainSchema and CARLStepSchema."""

    def test_step_schema_defaults(self) -> None:
        step = CARLStepSchema(number=1, title="Test")
        assert step.step_type == "llm"
        assert step.retry_max == 3
        assert step.dependencies == []

    def test_evaluation_criteria_accepts_list(self) -> None:
        # CARL's EvaluationStepConfig.criteria is list[str]; the planner emits a
        # list. This used to raise "Input should be a valid string".
        step = CARLStepSchema(
            number=2, title="Gate", step_type="evaluation",
            evaluation_criteria=["all fields present", "no nulls"],
        )
        assert step.evaluation_criteria == ["all fields present", "no nulls"]

    def test_evaluation_criteria_coerces_bare_string(self) -> None:
        step = CARLStepSchema(
            number=2, title="Gate", step_type="evaluation",
            evaluation_criteria="the forecast covers 7 days",
        )
        assert step.evaluation_criteria == ["the forecast covers 7 days"]

    def test_evaluation_criteria_empty_string_is_none(self) -> None:
        step = CARLStepSchema(
            number=2, title="Gate", step_type="evaluation",
            evaluation_criteria="   ",
        )
        assert step.evaluation_criteria is None

    def test_chain_schema_serialization(self) -> None:
        chain = CARLChainSchema(
            name="Test Chain",
            steps=[
                CARLStepSchema(
                    number=1,
                    title="First step",
                    aim="Do something",
                    reasoning_questions="What to do?",
                    step_context_queries=["keyword"],
                    stage_action="Analyze",
                    example_reasoning="Example",
                ),
            ],
        )
        d = chain.model_dump()
        assert d["name"] == "Test Chain"
        assert len(d["steps"]) == 1
        assert d["steps"][0]["step_type"] == "llm"

    def test_chain_schema_default_search_config(self) -> None:
        chain = CARLChainSchema()
        assert chain.search_config is not None
        assert chain.search_config["strategy"] == "substring"


class TestResearchSchemas:
    """Tests for research result schemas."""

    def test_memory_hit(self) -> None:
        hit = MemoryHit(entity_id="mem-123", name="test", score=0.8)
        assert hit.entity_id == "mem-123"
        assert hit.score == 0.8
        assert hit.abstract_structure == {}

    def test_memory_research_result_defaults(self) -> None:
        result = MemoryResearchResult()
        assert result.digest == ""
        assert result.hits_count == 0
        assert result.was_cold_start is True
        assert result.sub_queries_used == []

    def test_memory_research_result_values(self) -> None:
        result = MemoryResearchResult(
            digest="Some digest",
            hits_count=3,
            was_cold_start=False,
            sub_queries_used=["q1", "q2"],
        )
        assert result.digest == "Some digest"
        assert result.hits_count == 3
        assert result.was_cold_start is False

    def test_web_research_result_defaults(self) -> None:
        result = WebResearchResult()
        assert result.digest == ""
        assert result.sources == []
        assert result.was_used is False

    def test_web_research_result_values(self) -> None:
        result = WebResearchResult(
            digest="Web digest",
            sources=["https://example.com"],
            sub_queries_used=["q1"],
            was_used=True,
        )
        assert result.was_used is True
        assert len(result.sources) == 1
