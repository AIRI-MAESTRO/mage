"""Local registry of sandbox-validated generated tools (WP-9).

Synthesized tools land on disk under ``~/.mage/generated_tools/``; this index
records each validated tool so the :class:`CapabilityLookupAgent` can surface it
on the NEXT run — the planner then reuses the existing tool instead of paying to
re-synthesize it. This is the memory-free fallback for the capability registry
(``MemoryManager`` currently exposes no capability *write*).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# User-config dir — NOT the repo. Generated code is never committed.
DEFAULT_TOOLS_DIR = Path.home() / ".mage" / "generated_tools"
INDEX_NAME = "index.json"

# Serialize the index read-modify-write within this process (parallel generation
# runs many register_tool calls). Cross-process callers add an fcntl lock below.
_INDEX_LOCK = threading.Lock()


def _index_path(tools_dir: Path) -> Path:
    return tools_dir / INDEX_NAME


def load_tool_index(tools_dir: Path) -> list[dict[str, Any]]:
    """Return the registered tool entries (``[]`` when missing / unreadable)."""
    path = _index_path(tools_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — a corrupt index must never block generation
        logger.warning("generated-tool index unreadable (%s); ignoring", exc)
        return []
    return data if isinstance(data, list) else []


@contextlib.contextmanager
def _cross_process_lock(index_path: Path) -> Iterator[None]:
    """Best-effort cross-process exclusive lock via ``fcntl`` on a sidecar
    ``.lock`` file. A no-op where ``fcntl`` is unavailable (e.g. Windows) — the
    in-process :data:`_INDEX_LOCK` still serializes the common case."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    lock_path = index_path.with_name(index_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w", encoding="utf-8")  # noqa: SIM115 — released in finally
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(Exception):
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write ``data`` as JSON to ``path`` atomically: a temp file in the SAME
    directory, fsync, then :func:`os.replace` — so a concurrent reader never sees
    a half-written index. The temp file is cleaned up on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)  # atomic on POSIX + Windows
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def register_tool(tools_dir: Path, entry: dict[str, Any]) -> None:
    """Upsert one tool ``entry`` (keyed by ``name``) into the local index.

    Concurrency-safe: the whole read-modify-write runs under an in-process lock
    plus a best-effort cross-process file lock, and the index is replaced
    atomically — so parallel generation can't corrupt the file or lose entries.

    Best-effort on errors: a write failure is logged, never raised — registration
    must not fail a validated tool.
    """
    if not entry.get("name"):
        return
    tools_dir.mkdir(parents=True, exist_ok=True)
    path = _index_path(tools_dir)
    try:
        with _INDEX_LOCK, _cross_process_lock(path):
            index = [e for e in load_tool_index(tools_dir) if e.get("name") != entry["name"]]
            index.append(entry)
            _atomic_write_json(path, index)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not write generated-tool index (%s)", exc)


def find_tool(tools_dir: Path, *, name: str | None = None, need: str | None = None) -> dict[str, Any] | None:
    """Find a registered tool by exact ``name``, else by keyword overlap with ``need``.

    Used before synthesizing a new tool so an equivalent one already on disk is
    reused. Returns ``None`` when nothing matches.
    """
    index = load_tool_index(tools_dir)
    if name:
        for e in index:
            if e.get("name") == name:
                return e
    if need:
        need_tokens = {w for w in need.lower().split() if len(w) >= 4}
        best: dict[str, Any] | None = None
        best_score = 0
        for e in index:
            hay = f"{e.get('name', '')} {e.get('description', '')} {e.get('signature', '')}".lower()
            score = sum(1 for t in need_tokens if t in hay)
            if score > best_score:
                best, best_score = e, score
        if best_score >= 2:  # need at least two content words to overlap
            return best
    return None
