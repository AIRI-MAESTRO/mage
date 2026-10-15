"""Tests for CARLStepSchema with heterogeneous step types."""

from __future__ import annotations

from mmar_mage.schemas import CARLChainSchema, CARLStepSchema


class TestCARLStepSchemaStepTypes:
    """Tests for CARLStepSchema with all step types."""

    def test_llm_step(self) -> None:
        step = CARLStepSchema(
            step_type="llm",
            number=1,
            title="Analyze data",
            aim="Analyze the dataset",
            reasoning_questions="What patterns exist?",
            step_context_queries=["patterns", "trends"],
            stage_action="Analyze data patterns",
            example_reasoning="Identifying trends reveals insights",
        )
        assert step.step_type == "llm"
        assert step.aim == "Analyze the dataset"

    def test_tool_step(self) -> None:
        step = CARLStepSchema(
            step_type="tool",
            number=1,
            title="Fetch data",
            aim="Retrieve external data",
            tool_name="fetch_api",
            tool_description="Fetches data from external API",
            tool_input_mapping={"query": "$history[-1]"},
            tool_output_key="api_result",
            tool_timeout=30.0,
        )
        assert step.step_type == "tool"
        assert step.tool_name == "fetch_api"
        assert step.tool_input_mapping == {"query": "$history[-1]"}
        assert step.tool_output_key == "api_result"
        assert step.tool_timeout == 30.0

    def test_mcp_step(self) -> None:
        step = CARLStepSchema(
            step_type="mcp",
            number=1,
            title="Query database",
            aim="Execute SQL query",
            mcp_server_name="postgres",
            mcp_server_transport="stdio",
            mcp_server_command="npx",
            mcp_server_args=["@modelcontextprotocol/server-postgres"],
            mcp_tool_name="query",
            mcp_arguments={"sql": "SELECT * FROM users"},
            mcp_timeout=60.0,
        )
        assert step.step_type == "mcp"
        assert step.mcp_server_name == "postgres"
        assert step.mcp_tool_name == "query"
        assert step.mcp_server_transport == "stdio"
        assert step.mcp_server_args == ["@modelcontextprotocol/server-postgres"]

    def test_eval_step(self) -> None:
        step = CARLStepSchema(
            step_type="eval",
            number=2,
            title="Evaluate quality",
            aim="Assess output quality",
            eval_target_steps=[1],
            eval_criteria="Completeness and accuracy",
            eval_on_fail="retry",
            dependencies=[1],
        )
        assert step.step_type == "eval"
        assert step.eval_target_steps == [1]
        assert step.eval_criteria == "Completeness and accuracy"
        assert step.eval_on_fail == "retry"

    def test_conditional_step(self) -> None:
        step = CARLStepSchema(
            step_type="conditional",
            number=3,
            title="Branch on quality",
            aim="Decide next action based on eval",
            condition_branches=[
                {"condition": "quality >= 0.8", "next_step": 5},
                {"condition": "quality < 0.8", "next_step": 4},
            ],
            condition_default_step=4,
            condition_context_key="eval_result",
            dependencies=[2],
        )
        assert step.step_type == "conditional"
        assert len(step.condition_branches) == 2
        assert step.condition_default_step == 4

    def test_aggregator_step(self) -> None:
        step = CARLStepSchema(
            step_type="aggregator",
            number=4,
            title="Merge results",
            aim="Combine parallel analyses",
            aggregate_from_steps=[2, 3],
            aggregation_strategy="merge",
            dependencies=[2, 3],
        )
        assert step.step_type == "aggregator"
        assert step.aggregate_from_steps == [2, 3]
        assert step.aggregation_strategy == "merge"

    def test_memory_step(self) -> None:
        step = CARLStepSchema(
            step_type="memory",
            number=1,
            title="Save results",
            aim="Persist analysis results",
            memory_operation="write",
            memory_key="analysis_result",
            memory_value_source="$history[-1]",
            memory_namespace="default",
        )
        assert step.step_type == "memory"
        assert step.memory_operation == "write"
        assert step.memory_key == "analysis_result"
        assert step.memory_namespace == "default"

    def test_transform_step(self) -> None:
        step = CARLStepSchema(
            step_type="transform",
            number=1,
            title="Extract fields",
            aim="Extract specific fields from data",
            transform_type="extract",
            transform_input_key="raw_data",
            transform_output_format="json",
            transform_expression="$.results[*].name",
        )
        assert step.step_type == "transform"
        assert step.transform_type == "extract"
        assert step.transform_input_key == "raw_data"
        assert step.transform_expression == "$.results[*].name"

    def test_type_specific_fields_default_to_none(self) -> None:
        """Non-applicable type-specific fields default to None."""
        step = CARLStepSchema(number=1, title="Test", step_type="llm")
        assert step.tool_name is None
        assert step.mcp_server_name is None
        assert step.eval_target_steps is None
        assert step.condition_branches is None
        assert step.aggregate_from_steps is None
        assert step.memory_operation is None
        assert step.transform_type is None

    def test_chain_with_mixed_step_types(self) -> None:
        """CARLChainSchema can hold steps of different types."""
        chain = CARLChainSchema(
            name="Mixed Chain",
            steps=[
                CARLStepSchema(number=1, title="Fetch data", step_type="tool", tool_name="api_fetch"),
                CARLStepSchema(number=2, title="Analyze", step_type="llm", aim="Analyze fetched data"),
                CARLStepSchema(
                    number=3,
                    title="Evaluate",
                    step_type="eval",
                    eval_target_steps=[2],
                    eval_criteria="accuracy",
                    dependencies=[2],
                ),
                CARLStepSchema(
                    number=4,
                    title="Save",
                    step_type="memory",
                    memory_operation="write",
                    memory_key="result",
                    dependencies=[3],
                ),
            ],
        )
        d = chain.model_dump(exclude_none=True)
        assert len(d["steps"]) == 4
        assert d["steps"][0]["step_type"] == "tool"
        assert d["steps"][1]["step_type"] == "llm"
        assert d["steps"][2]["step_type"] == "eval"
        assert d["steps"][3]["step_type"] == "memory"

    def test_serialization_excludes_none(self) -> None:
        """model_dump(exclude_none=True) omits unused type-specific fields."""
        step = CARLStepSchema(
            number=1,
            title="Tool step",
            step_type="tool",
            tool_name="fetch",
            tool_input_mapping={"q": "value"},
        )
        d = step.model_dump(exclude_none=True)
        assert "tool_name" in d
        assert "mcp_server_name" not in d
        assert "eval_target_steps" not in d
        assert "memory_operation" not in d
