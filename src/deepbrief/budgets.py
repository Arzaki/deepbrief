"""Per-run budgets. Abort gracefully with a partial result + clear reason when exceeded."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budgets:
    max_llm_calls: int = 40
    max_tool_calls: int = 30
    max_tokens: int = 60000
    max_wall_seconds: int = 600
    started_at: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def start(self) -> None:
        self.started_at = time.time()

    def check(self) -> str | None:
        with self._lock:
            if self.llm_calls >= self.max_llm_calls:
                return f"llm_call_budget_exceeded({self.llm_calls}>={self.max_llm_calls})"
            if self.tool_calls >= self.max_tool_calls:
                return f"tool_call_budget_exceeded({self.tool_calls}>={self.max_tool_calls})"
            if self.tokens >= self.max_tokens:
                return f"token_budget_exceeded({self.tokens}>={self.max_tokens})"
            if self.started_at and (time.time() - self.started_at) >= self.max_wall_seconds:
                return f"wall_clock_exceeded({self.max_wall_seconds}s)"
            return None

    def register_llm(self, tokens: int) -> None:
        with self._lock:
            self.llm_calls += 1
            self.tokens += tokens

    def register_tool(self) -> None:
        with self._lock:
            self.tool_calls += 1
