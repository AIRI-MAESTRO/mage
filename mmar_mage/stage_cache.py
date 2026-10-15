"""T42 — content-addressable cache for deterministic deep-mode stages.

domain analysis / topology selection / capability lookup recompute the same
result for the same ``(query, domain)`` within a session and across pipeline
retries (visible in runs 731/773, where every stage ran twice per retry). This is
a small in-memory cache keyed by a hash of the stage name + its inputs — the
shared mechanism Layer A of the evolve design also builds on.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


class StageCache:
    """In-memory cache keyed by ``sha256(stage + canonical-JSON(payload))``.

    Session-scoped (one per generator). The generator stores + returns deep
    copies, so a cached value can never be mutated in place by a later stage.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(stage: str, payload: Any) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return f"{stage}:{hashlib.sha256(blob.encode('utf-8')).hexdigest()[:16]}"

    def get(self, stage: str, payload: Any) -> Any | None:
        cached = self._store.get(self.key(stage, payload), _MISS)
        if cached is _MISS:
            self.misses += 1
            return None
        self.hits += 1
        return cached

    def put(self, stage: str, payload: Any, value: Any) -> None:
        self._store[self.key(stage, payload)] = value

    def clear(self) -> None:
        self._store.clear()
        self.hits = self.misses = 0


_MISS = object()  # sentinel so a cached ``None`` is distinguishable from a miss
