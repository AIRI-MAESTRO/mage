"""Tests for CodeGenerator — Python code generation for each step type."""

from __future__ import annotations

from mmar_mage.code_generator import CodeGenerator
from mmar_mage.schemas import MAGEConfig


class TestCodeGenerator:
    """Tests for CodeGenerator.generate()."""

    def _config(self) -> MAGEConfig:
        return MAGEConfig(output_format="python", api_key="test")

    def test_generates_valid_python(self) -> None:
        chain_dict = {
            "name": "Test Chain",
            "max_workers": 2,
            "timeout": 300.0,
            "steps": [
                {
                    "step_type": "llm",
                    "number": 1,
                    "title": "Analyze",
                    "aim": "Analyze the data",
                    "reasoning_questions": "What patterns?",
                    "step_context_queries": ["patterns"],
                    "stage_action": "Analyze",
                    "example_reasoning": "Example",
                    "dependencies": [],
                },
            ],
        }
        code = CodeGenerator.generate(chain_dict, "Test query", self._config())
        assert "from mmar_carl import" in code
        assert "ChainBuilder" in code
        assert "builder.add_step(" in code
        assert 'if __name__ == "__main__":' in code
        compile(code, "<test>", "exec")

    def test_llm_step(self) -> None:
        chain_dict = {
            "name": "LLM Chain",
            "steps": [
                {
                    "step_type": "llm",
                    "number": 1,
                    "title": "Think",
                    "aim": "Reason about the problem",
                    "reasoning_questions": "What is the core issue?",
                    "stage_action": "Analyze",
                    "example_reasoning": "The core issue is...",
                    "dependencies": [],
                    "step_context_queries": [],
                },
            ],
        }
        code = CodeGenerator.generate(chain_dict, "Test", self._config())
        assert "builder.add_step(" in code
        assert "aim=" in code

    def test_tool_step(self) -> None:
        chain_dict = {
            "name": "Tool Chain",
            "steps": [
                {
                    "step_type": "tool",
                    "number": 1,
                    "title": "Fetch",
                    "tool_name": "api_fetch",
                    "tool_input_mapping": {"q": "query"},
                    "tool_description": "Fetches data",
                    "tool_timeout": 30.0,
                    "dependencies": [],
                },
            ],
        }
        code = CodeGenerator.generate(chain_dict, "Test", self._config())
        assert "builder.add_tool_step(" in code
        assert "tool_name='api_fetch'" in code

    def test_mcp_step(self) -> None:
        chain_dict = {
            "name": "MCP Chain",
            "steps": [
                {
                    "step_type": "mcp",
                    "number": 1,
                    "title": "Query DB",
                    "mcp_server_name": "postgres",
                    "mcp_tool_name": "query",
                    "mcp_server_command": "npx",
                    "mcp_server_args": ["@mcp/postgres"],
                    "mcp_arguments": {"sql": "SELECT 1"},
                    "dependencies": [],
                },
            ],
        }
        code = CodeGenerator.generate(chain_dict, "Test", self._config())
        assert "builder.add_mcp_step(" in code
        assert "server_name='postgres'" in code

    def test_memory_step(self) -> None:
        chain_dict = {
            "name": "Memory Chain",
            "steps": [
                {
                    "step_type": "memory",
                    "number": 1,
                    "title": "Save",
                    "memory_operation": "write",
                    "memory_key": "result",
                    "memory_value_source": "$history[-1]",
                    "dependencies": [],
                },
            ],
        }
        code = CodeGenerator.generate(chain_dict, "Test", self._config())
        assert "builder.add_memory_step(" in code
        assert "operation='write'" in code

    def test_conditional_step(self) -> None:
        chain_dict = {
            "name": "Conditional Chain",
            "steps": [
                {
                    "step_type": "conditional",
                    "number": 1,
                    "title": "Branch",
                    "condition_branches": [
                        {"condition": "score > 0.8", "next_step": 3},
                    ],
                    "condition_default_step": 2,
                    "dependencies": [],
                },
            ],
        }
        code = CodeGenerator.generate(chain_dict, "Test", self._config())
        assert "builder.add_conditional_step(" in code
        assert "branches=" in code

    def test_transform_step(self) -> None:
        chain_dict = {
            "name": "Transform Chain",
            "steps": [
                {
                    "step_type": "transform",
                    "number": 1,
                    "title": "Extract",
                    "transform_type": "extract",
                    "transform_input_key": "raw",
                    "transform_expression": "$.data",
                    "dependencies": [],
                },
            ],
        }
        code = CodeGenerator.generate(chain_dict, "Test", self._config())
        assert "builder.add_transform_step(" in code
        assert "transform_type='extract'" in code

    def test_eval_step(self) -> None:
        chain_dict = {
            "name": "Eval Chain",
            "steps": [
                {
                    "step_type": "eval",
                    "number": 1,
                    "title": "Evaluate",
                    "aim": "Check quality",
                    "reasoning_questions": "Is output good?",
                    "stage_action": "Evaluate",
                    "example_reasoning": "Quality check",
                    "step_context_queries": [],
                    "dependencies": [],
                },
            ],
        }
        code = CodeGenerator.generate(chain_dict, "Test", self._config())
        assert "# eval step maps to LLM step with self-critic config" in code
        assert "builder.add_step(" in code

    def test_aggregator_step(self) -> None:
        chain_dict = {
            "name": "Aggregator Chain",
            "steps": [
                {
                    "step_type": "aggregator",
                    "number": 1,
                    "title": "Merge",
                    "aggregation_strategy": "merge",
                    "dependencies": [],
                },
            ],
        }
        code = CodeGenerator.generate(chain_dict, "Test", self._config())
        assert "# aggregator step maps to transform step with aggregate type" in code
        assert "builder.add_transform_step(" in code

    def test_mixed_chain(self) -> None:
        chain_dict = {
            "name": "Mixed Chain",
            "max_workers": 3,
            "timeout": 600.0,
            "search_config": {"strategy": "substring"},
            "steps": [
                {
                    "step_type": "tool",
                    "number": 1,
                    "title": "Fetch",
                    "tool_name": "api",
                    "tool_input_mapping": {},
                    "dependencies": [],
                },
                {
                    "step_type": "llm",
                    "number": 2,
                    "title": "Analyze",
                    "aim": "Analyze",
                    "reasoning_questions": "What?",
                    "stage_action": "Analyze",
                    "example_reasoning": "Ex",
                    "step_context_queries": [],
                    "dependencies": [1],
                },
                {
                    "step_type": "eval",
                    "number": 3,
                    "title": "Check",
                    "aim": "Check",
                    "reasoning_questions": "Good?",
                    "stage_action": "Check",
                    "example_reasoning": "Check",
                    "step_context_queries": [],
                    "dependencies": [2],
                },
            ],
        }
        code = CodeGenerator.generate(chain_dict, "Mixed test", self._config())
        assert "builder.add_tool_step(" in code
        assert "builder.add_step(" in code
        assert "with_max_workers(3)" in code
        assert "with_timeout(600.0)" in code
        compile(code, "<test>", "exec")

    def test_search_config_included(self) -> None:
        chain_dict = {
            "name": "Test",
            "search_config": {"strategy": "substring", "config": {}},
            "steps": [
                {
                    "step_type": "llm",
                    "number": 1,
                    "title": "Step",
                    "aim": "Do",
                    "reasoning_questions": "",
                    "stage_action": "",
                    "example_reasoning": "",
                    "step_context_queries": [],
                    "dependencies": [],
                },
            ],
        }
        code = CodeGenerator.generate(chain_dict, "Test", self._config())
        assert "with_search_config(" in code
