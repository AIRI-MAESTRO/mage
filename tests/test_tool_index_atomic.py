"""T28 — generated-tool index writes are atomic + concurrency-safe."""

from __future__ import annotations

import threading
from pathlib import Path

from mmar_mage.generated_tool_index import load_tool_index, register_tool


def test_register_and_load_roundtrip(tmp_path: Path) -> None:
    register_tool(tmp_path, {"name": "alpha", "description": "a"})
    register_tool(tmp_path, {"name": "beta", "description": "b"})
    assert {e["name"] for e in load_tool_index(tmp_path)} == {"alpha", "beta"}


def test_register_tool_upserts_by_name(tmp_path: Path) -> None:
    register_tool(tmp_path, {"name": "t", "description": "v1"})
    register_tool(tmp_path, {"name": "t", "description": "v2"})
    index = load_tool_index(tmp_path)
    assert len(index) == 1
    assert index[0]["description"] == "v2"


def test_atomic_write_leaves_no_temp_files(tmp_path: Path) -> None:
    register_tool(tmp_path, {"name": "a"})
    register_tool(tmp_path, {"name": "b"})
    assert not list(tmp_path.glob("*.tmp"))  # os.replace consumed the temp file


def test_concurrent_registration_loses_no_entries(tmp_path: Path) -> None:
    n = 24

    def reg(i: int) -> None:
        register_tool(tmp_path, {"name": f"tool_{i}", "description": f"d{i}"})

    threads = [threading.Thread(target=reg, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    index = load_tool_index(tmp_path)
    assert {e["name"] for e in index} == {f"tool_{i}" for i in range(n)}
    assert len(index) == n  # the lock prevented lost updates under concurrency
