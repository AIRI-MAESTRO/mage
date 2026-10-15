"""
Code generator for ChainBuilder fluent API Python scripts.

Generates self-contained Python scripts that build and execute CARL chains
using the ChainBuilder fluent API, mapping each step type to the correct
builder method.
"""

from __future__ import annotations

import json
from typing import Any

from .schemas import MAGEConfig


class CodeGenerator:
    """Generate Python scripts using the ChainBuilder fluent API."""

    @staticmethod
    def generate(chain_dict: dict[str, Any], query: str, config: MAGEConfig) -> str:
        """Generate a complete Python script that builds and executes a CARL chain.

        Parameters
        ----------
        chain_dict : dict[str, Any]
            The CARL chain dictionary.
        query : str
            Original user query.
        config : MAGEConfig
            Generation configuration.

        Returns
        -------
        str
            Complete Python script as a string.
        """
        lines: list[str] = [
            '"""Auto-generated CARL chain script."""',
            "",
            "from mmar_carl import ChainBuilder, ReasoningChain, ReasoningContext, Language",
            "",
            "",
            "def build_chain() -> ReasoningChain:",
            f'    """Build chain for: {query[:80]}"""',
            "    builder = ChainBuilder()",
            "",
        ]

        max_workers = chain_dict.get("max_workers", config.default_max_workers)
        timeout = chain_dict.get("timeout", config.default_timeout)

        lines.append(f"    builder.with_max_workers({max_workers})")
        lines.append(f"    builder.with_timeout({timeout})")

        search_config = chain_dict.get("search_config")
        if search_config:
            lines.append(f"    builder.with_search_config({json.dumps(search_config)})")

        lines.append("")

        steps = chain_dict.get("steps", [])
        for step in steps:
            step_lines = CodeGenerator._generate_step(step)
            lines.extend(f"    {line}" for line in step_lines)
            lines.append("")

        lines.append("    return builder.build()")
        lines.append("")
        lines.append("")
        lines.append('if __name__ == "__main__":')
        lines.append("    chain = build_chain()")
        lines.append("    context = ReasoningContext(")
        lines.append(f'        query="{query[:100]}",')
        lines.append("        language=Language.EN,")
        lines.append("    )")
        lines.append("    result = chain.execute(context)")
        lines.append('    print("Result:", result)')
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _generate_step(step: dict[str, Any]) -> list[str]:
        """Generate builder method call for a single step."""
        step_type = step.get("step_type", "llm")
        number = step.get("number", 1)
        title = step.get("title", "")
        deps = step.get("dependencies", [])

        if step_type == "tool":
            return CodeGenerator._gen_tool_step(step, number, title, deps)
        elif step_type == "mcp":
            return CodeGenerator._gen_mcp_step(step, number, title, deps)
        elif step_type == "memory":
            return CodeGenerator._gen_memory_step(step, number, title, deps)
        elif step_type == "conditional":
            return CodeGenerator._gen_conditional_step(step, number, title, deps)
        elif step_type == "transform":
            return CodeGenerator._gen_transform_step(step, number, title, deps)
        elif step_type == "eval":
            return CodeGenerator._gen_eval_step(step, number, title, deps)
        elif step_type == "aggregator":
            return CodeGenerator._gen_aggregator_step(step, number, title, deps)
        else:
            return CodeGenerator._gen_llm_step(step, number, title, deps)

    @staticmethod
    def _gen_llm_step(step: dict[str, Any], number: int, title: str, deps: list[int]) -> list[str]:
        aim = step.get("aim", "")
        rq = step.get("reasoning_questions", "")
        sa = step.get("stage_action", "")
        er = step.get("example_reasoning", "")
        scq = step.get("step_context_queries", [])
        lines = [
            "builder.add_step(",
            f"    number={number},",
            f"    title={title!r},",
            f"    aim={aim!r},",
            f"    reasoning_questions={rq!r},",
            f"    stage_action={sa!r},",
            f"    example_reasoning={er!r},",
            f"    dependencies={deps!r},",
            f"    step_context_queries={scq!r},",
            ")",
        ]
        return lines

    @staticmethod
    def _gen_tool_step(step: dict[str, Any], number: int, title: str, deps: list[int]) -> list[str]:
        tool_name = step.get("tool_name", "unknown")
        input_mapping = step.get("tool_input_mapping", {})
        tool_desc = step.get("tool_description")
        timeout = step.get("tool_timeout")
        lines = [
            "builder.add_tool_step(",
            f"    number={number},",
            f"    title={title!r},",
            f"    tool_name={tool_name!r},",
            f"    input_mapping={input_mapping!r},",
            f"    dependencies={deps!r},",
        ]
        # NOTE: ChainBuilder.add_tool_step() has no output_key parameter (unlike
        # add_memory_step's `namespace` / add_conditional_step's `condition_context_key`),
        # so the builder-codegen path leaves the tool output at CARL's default "result"
        # key. A custom tool_output_key is preserved by the JSON export path
        # (to_carl_compat maps tool_output_key -> output_key in CARL_FIELD_MAP).
        if tool_desc:
            lines.append(f"    tool_description={tool_desc!r},")
        if timeout:
            lines.append(f"    timeout={timeout},")
        lines.append(")")
        return lines

    @staticmethod
    def _gen_mcp_step(step: dict[str, Any], number: int, title: str, deps: list[int]) -> list[str]:
        server_name = step.get("mcp_server_name", "unknown")
        tool_name = step.get("mcp_tool_name", "unknown")
        command = step.get("mcp_server_command")
        args = step.get("mcp_server_args")
        arguments = step.get("mcp_arguments")
        argument_mapping = step.get("mcp_argument_mapping")
        lines = [
            "builder.add_mcp_step(",
            f"    number={number},",
            f"    title={title!r},",
            f"    server_name={server_name!r},",
            f"    tool_name={tool_name!r},",
        ]
        if command:
            lines.append(f"    command={command!r},")
        if args:
            lines.append(f"    args={args!r},")
        if arguments:
            lines.append(f"    arguments={arguments!r},")
        if argument_mapping:
            lines.append(f"    argument_mapping={argument_mapping!r},")
        lines.append(f"    dependencies={deps!r},")
        lines.append(")")
        return lines

    @staticmethod
    def _gen_memory_step(step: dict[str, Any], number: int, title: str, deps: list[int]) -> list[str]:
        operation = step.get("memory_operation", "read")
        memory_key = step.get("memory_key", "")
        value_source = step.get("memory_value_source")
        namespace = step.get("memory_namespace")
        lines = [
            "builder.add_memory_step(",
            f"    number={number},",
            f"    title={title!r},",
            f"    operation={operation!r},",
            f"    memory_key={memory_key!r},",
        ]
        if value_source:
            lines.append(f"    value_source={value_source!r},")
        if namespace:  # T31: was dropped — reads/writes would hit the default namespace
            lines.append(f"    namespace={namespace!r},")
        lines.append(f"    dependencies={deps!r},")
        lines.append(")")
        return lines

    @staticmethod
    def _gen_conditional_step(step: dict[str, Any], number: int, title: str, deps: list[int]) -> list[str]:
        branches = step.get("condition_branches", [])
        default_step = step.get("condition_default_step")
        context_key = step.get("condition_context_key")
        branch_tuples = [(b.get("condition", ""), b.get("next_step", 0)) for b in branches]
        lines = [
            "builder.add_conditional_step(",
            f"    number={number},",
            f"    title={title!r},",
            f"    branches={branch_tuples!r},",
        ]
        if default_step is not None:
            lines.append(f"    default_step={default_step},")
        if context_key:  # T31: was dropped — the branch conditions evaluate against it
            lines.append(f"    condition_context_key={context_key!r},")
        lines.append(f"    dependencies={deps!r},")
        lines.append(")")
        return lines

    @staticmethod
    def _gen_transform_step(step: dict[str, Any], number: int, title: str, deps: list[int]) -> list[str]:
        transform_type = step.get("transform_type", "extract")
        input_key = step.get("transform_input_key", "")
        output_format = step.get("transform_output_format")
        expression = step.get("transform_expression")
        lines = [
            "builder.add_transform_step(",
            f"    number={number},",
            f"    title={title!r},",
            f"    transform_type={transform_type!r},",
            f"    input_key={input_key!r},",
        ]
        if output_format:
            lines.append(f"    output_format={output_format!r},")
        if expression:
            lines.append(f"    expression={expression!r},")
        lines.append(f"    dependencies={deps!r},")
        lines.append(")")
        return lines

    @staticmethod
    def _gen_eval_step(step: dict[str, Any], number: int, title: str, deps: list[int]) -> list[str]:
        aim = step.get("aim", "")
        rq = step.get("reasoning_questions", "")
        sa = step.get("stage_action", "")
        er = step.get("example_reasoning", "")
        scq = step.get("step_context_queries", [])
        lines = [
            "# eval step maps to LLM step with self-critic config",
            "builder.add_step(",
            f"    number={number},",
            f"    title={title!r},",
            f"    aim={aim!r},",
            f"    reasoning_questions={rq!r},",
            f"    stage_action={sa!r},",
            f"    example_reasoning={er!r},",
            f"    dependencies={deps!r},",
            f"    step_context_queries={scq!r},",
            ")",
        ]
        return lines

    @staticmethod
    def _gen_aggregator_step(step: dict[str, Any], number: int, title: str, deps: list[int]) -> list[str]:
        strategy = step.get("aggregation_strategy", "concat")
        input_key = step.get("transform_input_key", "")
        lines = [
            "# aggregator step maps to transform step with aggregate type",
            "builder.add_transform_step(",
            f"    number={number},",
            f"    title={title!r},",
            "    transform_type='aggregate',",
            f"    input_key={input_key!r},",
        ]
        if strategy != "concat":
            lines.append(f"    expression={strategy!r},")
        lines.append(f"    dependencies={deps!r},")
        lines.append(")")
        return lines
