"""Writer: findings -> Report. Only uses supplied claims; every factual sentence needs [Sx].

Long reports despite the per-call output cap: one call writes title+summary,
then one call per sub-question writes a full section (in parallel). Four
sections of ~750 words beats a single call's ceiling without adding outside facts.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..llm import BaseProvider
from ..schemas import Findings, Report, ReportHead, ReportSection
from .loop import complete_json_with_retry

PROMPT = Path(__file__).resolve().parents[3] / "prompts" / "writer.md"


def findings_to_text(findings: list[Findings], quote_chars: int = 200) -> str:
    parts = []
    for f in findings:
        for c in f.claims:
            parts.append(f"- [{c.source_id}] {c.text} (quote: {c.supporting_quote[:quote_chars]})")
    return "\n".join(parts)


def _claims_condensed(findings: list[Findings]) -> str:
    return "\n".join(f"- [{c.source_id}] {c.text}" for f in findings for c in f.claims)


def run_writer(
    findings: list[Findings],
    llm: BaseProvider,
    model: str,
    trace=None,
    temperature: float = 0.0,
    seed: int | None = None,
    max_tokens: int = 1000,
    feedback: str = "",
    max_workers: int = 3,
) -> Report:
    try:
        system = PROMPT.read_text(encoding="utf-8")
    except OSError:
        system = "Write cited report as strict JSON."
    common = {
        "llm": llm,
        "model": model,
        "trace": trace,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
    }
    if not any(f.claims for f in findings):
        # Honest-empty fast path: nothing to sectionize; one call, same as before.
        user = "Write the report from these findings (use ONLY these claims):\n(empty — no verified claims)"
        if feedback:
            user += f"\n\nReviewer feedback to address:\n{feedback}"
        return complete_json_with_retry(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            pydantic_model=Report,
            agent_name="writer",
            **common,
        )
    # 1) Title + summary from all claims (condensed — quotes not needed here).
    head_user = (
        "Write ONLY the report title and a 3-5 sentence summary covering the claims below "
        "(use ONLY these claims):\n" + _claims_condensed(findings)
    )
    if feedback:
        head_user += f"\n\nReviewer feedback to address:\n{feedback}"
    head = complete_json_with_retry(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": head_user}],
        pydantic_model=ReportHead,
        agent_name="writer-head",
        **common,
    )

    # 2) One full section per sub-question with claims (parallel: independent calls).
    groups = [f for f in findings if f.claims]

    def _section(f: Findings) -> ReportSection:
        sec_user = (
            f"Write ONE thorough report section for research findings [{f.sub_question_id}] "
            "(3-6 substantive paragraphs covering EVERY claim below; compare/contrast where "
            "claims relate; every factual sentence ends with its [Sx] marker; use ONLY these "
            "claims):\n" + findings_to_text([f], quote_chars=800)
        )
        if feedback:
            sec_user += f"\n\nReviewer feedback to address:\n{feedback}"
        return complete_json_with_retry(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": sec_user}],
            pydantic_model=ReportSection,
            agent_name="writer-section",
            **common,
        )

    if max_workers <= 1:
        sections = [_section(f) for f in groups]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            sections = list(pool.map(_section, groups))
    citations = sorted({c.source_id for f in findings for c in f.claims} | {s.id for f in findings for s in f.sources})
    return Report(title=head.title, summary=head.summary, sections=sections, citations=citations)
