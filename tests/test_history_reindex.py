"""reindex_history_refs — MAGE step-NUMBER $history[N] -> CARL history POSITION.

MAGE plans $history refs by step number; mmar_carl resolves context.history
positionally, so an un-translated ref to a step whose number exceeds the run's
history depth raised "list index out of range". These lock the translation.
"""

from __future__ import annotations

from mmar_mage.carl_export import reindex_history_refs


def test_offbyone_two_step_chain():
    # step 2 reads step 1 — number 1, but at step 2 history has just [r1] (pos 0).
    chain = {
        "steps": [
            {"number": 1, "step_type": "tool", "dependencies": [], "step_config": {"tool_name": "x"}},
            {"number": 2, "step_type": "llm", "dependencies": [1],
             "step_config": {"input_source": "$history[1]"}},
        ]
    }
    reindex_history_refs(chain)
    assert chain["steps"][1]["step_config"]["input_source"] == "$history[0]"


def test_multi_ref_linear_chain():
    # The weather shape: step 4 (transform) reads step 2 (pos 1) + step 3 (pos 2);
    # step 5 reads step 4 (pos 3). The bare $history[3]/$history[4] would crash.
    chain = {
        "steps": [
            {"number": 1, "step_type": "llm", "dependencies": []},
            {"number": 2, "step_type": "tool", "dependencies": [1], "step_config": {"tool_name": "dt"}},
            {"number": 3, "step_type": "tool", "dependencies": [1],
             "step_config": {"tool_name": "web_search", "input_mapping": {"query": "$outer_context"}}},
            {"number": 4, "step_type": "transform", "dependencies": [2, 3], "step_config": {
                "input_key": "$history[3]",
                "expression": '{"date": "$history[2]", "temp": "$history[3].temperature"}',
            }},
            {"number": 5, "step_type": "evaluation", "dependencies": [4],
             "step_config": {"condition_context_key": "$history[4]"}},
        ]
    }
    reindex_history_refs(chain)
    s4 = chain["steps"][3]["step_config"]
    assert s4["input_key"] == "$history[2]"
    assert s4["expression"] == '{"date": "$history[1]", "temp": "$history[2].temperature"}'
    assert chain["steps"][4]["step_config"]["condition_context_key"] == "$history[3]"
    # dependencies stay as step numbers (they wire the DAG, not history positions)
    assert chain["steps"][3]["dependencies"] == [2, 3]
    # placeholders that aren't $history[N] are left alone
    assert chain["steps"][2]["step_config"]["input_mapping"]["query"] == "$outer_context"


def test_negative_and_unknown_refs_untouched():
    chain = {
        "steps": [
            {"number": 1, "step_type": "tool", "dependencies": []},
            {"number": 2, "step_type": "llm", "dependencies": [1],
             "step_config": {"a": "$history[-1]", "b": "$history[0]", "c": "$history[9]"}},
        ]
    }
    reindex_history_refs(chain)
    cfg = chain["steps"][1]["step_config"]
    assert cfg["a"] == "$history[-1]"  # positional-from-end fallback — leave it
    assert cfg["b"] == "$history[0]"   # 0 isn't a step number — leave it
    assert cfg["c"] == "$history[9]"   # no step 9 — leave it (can't map)


def test_single_step_is_noop():
    chain = {"steps": [{"number": 1, "step_type": "llm", "dependencies": [],
                        "step_config": {"k": "$history[1]"}}]}
    out = reindex_history_refs(chain)
    assert out["steps"][0]["step_config"]["k"] == "$history[1]"  # <2 steps → untouched


def test_diamond_dag_positions():
    # 1 -> {2,3} -> 4 : batch order [1],[2,3],[4] → positions 1:0, 2:1, 3:2, 4:3
    chain = {
        "steps": [
            {"number": 1, "step_type": "llm", "dependencies": []},
            {"number": 2, "step_type": "tool", "dependencies": [1], "step_config": {"tool_name": "a"}},
            {"number": 3, "step_type": "tool", "dependencies": [1], "step_config": {"tool_name": "b"}},
            {"number": 4, "step_type": "transform", "dependencies": [2, 3],
             "step_config": {"expression": "$history[2] + $history[3]"}},
        ]
    }
    reindex_history_refs(chain)
    assert chain["steps"][3]["step_config"]["expression"] == "$history[1] + $history[2]"
