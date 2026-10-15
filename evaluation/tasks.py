"""
Benchmark task definitions and seed task registry.

Each benchmark has a set of representative seed tasks that are used during
evaluation to generate CARL chains and baseline prompts.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BenchmarkTask(BaseModel):
    """A single evaluation task.

    Parameters
    ----------
    benchmark : str
        Benchmark name (e.g. ``"hover"``, ``"hotpotqa"``).
    task_id : str
        Unique task identifier (e.g. ``"hover_001"``).
    query : str
        The query to generate a chain for.
    category : str
        Optional sub-category within the benchmark.
    metadata : dict[str, Any]
        Extra benchmark-specific information.
    """

    benchmark: str
    task_id: str
    query: str
    category: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Seed tasks registry
# ---------------------------------------------------------------------------

BENCHMARK_TASKS: dict[str, list[BenchmarkTask]] = {
    "hover": [
        BenchmarkTask(
            benchmark="hover",
            task_id="hover_001",
            query=(
                "Verify whether the director of Inception also directed a film starring "
                "Leonardo DiCaprio that won the Academy Award for Best Picture."
            ),
            category="multi_hop_verification",
        ),
        BenchmarkTask(
            benchmark="hover",
            task_id="hover_002",
            query=(
                "Confirm that the country where the 2016 Summer Olympics were held is "
                "also the largest country in South America by area."
            ),
            category="multi_hop_verification",
        ),
        BenchmarkTask(
            benchmark="hover",
            task_id="hover_003",
            query=(
                "Verify that the author of A Brief History of Time held the same "
                "academic position at Cambridge that Isaac Newton once occupied."
            ),
            category="multi_hop_verification",
        ),
        BenchmarkTask(
            benchmark="hover",
            task_id="hover_004",
            query=(
                "Determine whether the company that created the iPhone also released "
                "the first commercially successful personal computer."
            ),
            category="multi_hop_verification",
        ),
        BenchmarkTask(
            benchmark="hover",
            task_id="hover_005",
            query=(
                "Check if the element with atomic number 79 is the same metal used in "
                "the Nobel Prize medals awarded in Stockholm."
            ),
            category="multi_hop_verification",
        ),
    ],
    "ifbench": [
        BenchmarkTask(
            benchmark="ifbench",
            task_id="ifbench_001",
            query=(
                "Write a 200-word summary of quantum computing. Use exactly 3 paragraphs. "
                "Include at least 2 technical terms. Do not use the word 'revolutionary'."
            ),
            category="constrained_generation",
        ),
        BenchmarkTask(
            benchmark="ifbench",
            task_id="ifbench_002",
            query=(
                "List 5 benefits of renewable energy. Each benefit must start with a verb. "
                "The total response must be under 150 words. Use bullet points."
            ),
            category="constrained_generation",
        ),
        BenchmarkTask(
            benchmark="ifbench",
            task_id="ifbench_003",
            query=(
                "Explain the process of photosynthesis in exactly 4 sentences. "
                "Each sentence must contain at least one scientific term. "
                "Do not use the passive voice."
            ),
            category="constrained_generation",
        ),
        BenchmarkTask(
            benchmark="ifbench",
            task_id="ifbench_004",
            query=(
                "Write a recipe for chocolate cake. Include exactly 7 ingredients. "
                "Each instruction step must be one sentence. Use metric units only."
            ),
            category="constrained_generation",
        ),
        BenchmarkTask(
            benchmark="ifbench",
            task_id="ifbench_005",
            query=(
                "Describe three differences between Python and JavaScript. "
                "Use a numbered list. Each item must be under 30 words. "
                "Do not mention 'web' or 'browser'."
            ),
            category="constrained_generation",
        ),
    ],
    "hotpotqa": [
        BenchmarkTask(
            benchmark="hotpotqa",
            task_id="hotpotqa_001",
            query=(
                "What is the birthplace of the author who wrote the novel that inspired "
                "the 1994 film The Shawshank Redemption?"
            ),
            category="multi_hop_qa",
        ),
        BenchmarkTask(
            benchmark="hotpotqa",
            task_id="hotpotqa_002",
            query=(
                "Which university did the physicist who formulated the general theory of "
                "relativity attend for his undergraduate studies?"
            ),
            category="multi_hop_qa",
        ),
        BenchmarkTask(
            benchmark="hotpotqa",
            task_id="hotpotqa_003",
            query=("What is the capital city of the country where the inventor of the World Wide Web was born?"),
            category="multi_hop_qa",
        ),
        BenchmarkTask(
            benchmark="hotpotqa",
            task_id="hotpotqa_004",
            query=(
                "In what year was the company founded that acquired the studio which "
                "produced the original Star Wars film?"
            ),
            category="multi_hop_qa",
        ),
        BenchmarkTask(
            benchmark="hotpotqa",
            task_id="hotpotqa_005",
            query=("What instrument did the composer of The Four Seasons primarily play?"),
            category="multi_hop_qa",
        ),
    ],
    "pupa": [
        BenchmarkTask(
            benchmark="pupa",
            task_id="pupa_001",
            query=(
                "A user who prefers concise answers and dislikes jargon asks: "
                "explain how a neural network learns from data."
            ),
            category="personalized_qa",
            metadata={"persona": "concise, no-jargon"},
        ),
        BenchmarkTask(
            benchmark="pupa",
            task_id="pupa_002",
            query=("A 10-year-old student asks: why does the Moon change shape every night?"),
            category="personalized_qa",
            metadata={"persona": "child, age 10"},
        ),
        BenchmarkTask(
            benchmark="pupa",
            task_id="pupa_003",
            query=(
                "An experienced software engineer asks: compare event-driven "
                "and request-response architectures for real-time data pipelines."
            ),
            category="personalized_qa",
            metadata={"persona": "senior engineer"},
        ),
    ],
    "livebench_math": [
        BenchmarkTask(
            benchmark="livebench_math",
            task_id="livebench_math_001",
            query=("Find all integer solutions to the equation x^3 - 6x^2 + 11x - 6 = 0."),
            category="algebra",
        ),
        BenchmarkTask(
            benchmark="livebench_math",
            task_id="livebench_math_002",
            query=(
                "A train travels from city A to city B at 60 km/h and returns at "
                "90 km/h. What is the average speed for the entire round trip?"
            ),
            category="arithmetic",
        ),
        BenchmarkTask(
            benchmark="livebench_math",
            task_id="livebench_math_003",
            query=("Compute the definite integral of sin(x) * cos(x) from 0 to pi/2."),
            category="calculus",
        ),
        BenchmarkTask(
            benchmark="livebench_math",
            task_id="livebench_math_004",
            query=(
                "Find the number of positive integers less than 1000 that are "
                "divisible by exactly two of the primes 2, 3, 5, 7."
            ),
            category="number_theory",
        ),
        BenchmarkTask(
            benchmark="livebench_math",
            task_id="livebench_math_005",
            query=(
                "A bag contains 5 red, 4 blue, and 3 green marbles. Three marbles "
                "are drawn without replacement. What is the probability that all "
                "three marbles are different colors? Express your answer as a "
                "simplified fraction."
            ),
            category="probability",
        ),
    ],
    "simpleqa": [
        BenchmarkTask(
            benchmark="simpleqa",
            task_id="simpleqa_001",
            query="What is the chemical formula for table salt?",
            category="factual",
        ),
        BenchmarkTask(
            benchmark="simpleqa",
            task_id="simpleqa_002",
            query="Who painted the Mona Lisa?",
            category="factual",
        ),
        BenchmarkTask(
            benchmark="simpleqa",
            task_id="simpleqa_003",
            query="What is the speed of light in meters per second?",
            category="factual",
        ),
    ],
    # T12 — composite live queries (the 09.06 run-096 class): each needs several
    # chained external calls and must stay recovery-ready — a tool failure retries
    # then replans, never masks as success. Mandatory regression cases.
    "live_composite": [
        BenchmarkTask(
            benchmark="live_composite",
            task_id="live_composite_001",
            query="What's the weather where my IP is located, and what should I wear?",
            category="composite_live",
            metadata={"min_atomic_calls": 3, "requires_tools": True, "expect_replan_ready": True},
        ),
        BenchmarkTask(
            benchmark="live_composite",
            task_id="live_composite_002",
            query=(
                "Look up the current price of Bitcoin from a market API and tell me "
                "whether it rose or fell since yesterday."
            ),
            category="composite_live",
            metadata={"min_atomic_calls": 2, "requires_tools": True, "expect_replan_on_failure": True},
        ),
    ],
}


def get_tasks(
    benchmarks: list[str],
    max_per_benchmark: int | None = None,
) -> list[BenchmarkTask]:
    """Filter and return benchmark tasks.

    Parameters
    ----------
    benchmarks : list[str]
        Benchmark names to include.
    max_per_benchmark : int | None
        Maximum number of tasks per benchmark.  ``None`` returns all.

    Returns
    -------
    list[BenchmarkTask]
        Filtered list of tasks.
    """
    tasks: list[BenchmarkTask] = []
    for name in benchmarks:
        bench_tasks = BENCHMARK_TASKS.get(name, [])
        if max_per_benchmark is not None:
            bench_tasks = bench_tasks[:max_per_benchmark]
        tasks.extend(bench_tasks)
    return tasks
