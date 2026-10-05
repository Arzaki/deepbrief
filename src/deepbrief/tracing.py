"""Structured per-run JSON trace: every LLM/tool call, tokens, latency, cost, cache hits."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .costs import estimate_cost


class RunTrace:
    def __init__(self, trace_dir: str, prompt_version: str, config_snapshot: dict[str, Any]):
        self.run_id = uuid.uuid4().hex[:12]
        self.trace_dir = Path(trace_dir)
        self.prompt_version = prompt_version
        self.config_snapshot = config_snapshot
        self.events: list[dict[str, Any]] = []
        self.started = time.time()
        self.totals = {
            "llm_calls": 0,
            "tool_calls": 0,
            "tokens": 0,
            "cost_usd": 0.0,
            "cache_hits": 0,
        }
        self._lock = threading.Lock()

    def log(self, kind: str, name: str, **fields: Any) -> None:
        evt = {"t": round(time.time() - self.started, 3), "kind": kind, "name": name, **fields}
        with self._lock:
            self.events.append(evt)
            if kind == "llm":
                self.totals["llm_calls"] += 1
                self.totals["tokens"] += int(fields.get("input_tokens", 0) + fields.get("output_tokens", 0))
                self.totals["cost_usd"] += float(fields.get("cost_usd", 0.0))
            elif kind == "tool":
                self.totals["tool_calls"] += 1
                if fields.get("cache_hit"):
                    self.totals["cache_hits"] += 1

    def log_llm(
        self,
        name: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_s: float,
        **extra: Any,
    ) -> None:
        cost = estimate_cost(model, input_tokens, output_tokens)
        self.log(
            "llm",
            name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=round(latency_s, 3),
            cost_usd=round(cost, 6),
            **extra,
        )

    def save(self) -> Path:
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.run_id,
            "prompt_version": self.prompt_version,
            "config": self.config_snapshot,
            "totals": self.totals,
            "wall_s": round(time.time() - self.started, 3),
            "events": self.events,
        }
        path = self.trace_dir / f"run_{self.run_id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
