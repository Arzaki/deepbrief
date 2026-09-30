"""Cost estimator driven by a configurable price table. All values APPROXIMATE."""

from __future__ import annotations

from .config import PRICE_TABLE


def estimate_cost(model: str, input_tokens: int, output_tokens: int, table: dict | None = None) -> float:
    table = table or PRICE_TABLE
    p = table.get(model, table.get("default", {"input": 0.0, "output": 0.0}))
    return (input_tokens / 1000.0) * p["input"] + (output_tokens / 1000.0) * p["output"]


def summarize_cost(model: str, input_tokens: int, output_tokens: int) -> dict:
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd_approx": round(estimate_cost(model, input_tokens, output_tokens), 6),
        "note": "approximate; see config.PRICE_TABLE snapshot date",
    }
