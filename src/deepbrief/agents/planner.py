"""Planner agent: question -> Plan (3-5 sub-questions), validated JSON, max 2 retries."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from ..llm import BaseProvider
from ..schemas import Plan
from .loop import parse_json_strict

PROMPT = Path(__file__).resolve().parents[3] / "prompts" / "planner.md"


def _prompt_text() -> str:
    try:
        return PROMPT.read_text(encoding="utf-8")
    except OSError:
        return "Decompose into 3-5 sub-questions. Return strict JSON."


def run_planner(
    question: str,
    llm: BaseProvider,
    model: str,
    trace=None,
    temperature: float = 0.0,
    seed: int | None = None,
) -> Plan:
    system = _prompt_text()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Research question: {question}"},
    ]
    last_err = ""
    for _ in range(3):  # initial + 2 retries
        if last_err:
            messages = messages[:2] + [
                {
                    "role": "user",
                    "content": f"Previous output failed validation: {last_err}. Return ONLY valid JSON.",
                }
            ]
        resp = llm.complete(messages, model, temperature=temperature, seed=seed)
        if trace:
            trace.log_llm("planner", model, resp.input_tokens, resp.output_tokens, resp.latency_s)
        try:
            return Plan.model_validate(parse_json_strict(resp.text))
        except (ValidationError, ValueError) as e:
            last_err = str(e)[:500]
    raise ValueError(f"planner failed validation after retries: {last_err}")
