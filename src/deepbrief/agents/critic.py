"""Critic: draft + findings -> Critique (pass | revise)."""

from __future__ import annotations

from pathlib import Path

from ..llm import BaseProvider
from ..schemas import Critique, Findings, Report
from .loop import parse_json_strict
from .writer import findings_to_text

PROMPT = Path(__file__).resolve().parents[3] / "prompts" / "critic.md"


def run_critic(
    report: Report,
    findings: list[Findings],
    llm: BaseProvider,
    model: str,
    trace=None,
    temperature: float = 0.0,
    seed: int | None = None,
) -> Critique:
    try:
        system = PROMPT.read_text(encoding="utf-8")
    except OSError:
        system = "Critique the draft. Return strict JSON."
    user = (
        f"Draft report:\n{report.model_dump_json(indent=2)}\n\nFindings:\n{findings_to_text(findings)}\n"
        "Return Critique JSON."
    )
    resp = llm.complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model,
        temperature=temperature,
        seed=seed,
    )
    if trace:
        trace.log_llm("critic", model, resp.input_tokens, resp.output_tokens, resp.latency_s)
    return Critique.model_validate(parse_json_strict(resp.text))
