"""Writer: findings -> Report. Only uses supplied claims; every factual sentence needs [Sx]."""

from __future__ import annotations

from pathlib import Path

from ..llm import BaseProvider
from ..schemas import Findings, Report
from .loop import parse_json_strict

PROMPT = Path(__file__).resolve().parents[3] / "prompts" / "writer.md"


def findings_to_text(findings: list[Findings]) -> str:
    parts = []
    for f in findings:
        for c in f.claims:
            parts.append(f"- [{c.source_id}] {c.text} (quote: {c.supporting_quote[:200]})")
    return "\n".join(parts)


def run_writer(
    findings: list[Findings],
    llm: BaseProvider,
    model: str,
    trace=None,
    temperature: float = 0.0,
    seed: int | None = None,
    feedback: str = "",
) -> Report:
    try:
        system = PROMPT.read_text(encoding="utf-8")
    except OSError:
        system = "Write cited report as strict JSON."
    user = f"Write the report from these findings (use ONLY these claims):\n{findings_to_text(findings)}"
    if feedback:
        user += f"\n\nReviewer feedback to address:\n{feedback}"
    resp = llm.complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model,
        temperature=temperature,
        seed=seed,
    )
    if trace:
        trace.log_llm("writer", model, resp.input_tokens, resp.output_tokens, resp.latency_s)
    return Report.model_validate(parse_json_strict(resp.text))
