"""SQLite-backed on-disk cache for search results and fetched pages."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path


def _key(kind: str, raw: str) -> str:
    return kind + ":" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, path: str, enabled: bool = True, ttl_s: int = 30 * 86400):
        self.enabled = enabled
        self.ttl_s = ttl_s
        self.path = Path(path)
        self.hits = 0
        self.misses = 0
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path))
            self._conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT, ts REAL)")
            self._conn.commit()
        else:
            self._conn = None  # type: ignore

    def get(self, kind: str, raw: str) -> str | None:
        if not self.enabled:
            self.misses += 1
            return None
        k = _key(kind, raw)
        row = self._conn.execute("SELECT v, ts FROM kv WHERE k=?", (k,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        v, ts = row
        if time.time() - ts > self.ttl_s:
            self.misses += 1
            return None
        self.hits += 1
        return v

    def put(self, kind: str, raw: str, value: str) -> None:
        if not self.enabled:
            return
        k = _key(kind, raw)
        self._conn.execute("REPLACE INTO kv (k, v, ts) VALUES (?, ?, ?)", (k, value, time.time()))
        self._conn.commit()

    def get_json(self, kind: str, raw: str):
        v = self.get(kind, raw)
        return json.loads(v) if v is not None else None

    def put_json(self, kind: str, raw: str, obj) -> None:
        self.put(kind, raw, json.dumps(obj))

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
