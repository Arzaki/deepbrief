"""Single-agent baseline: one loop with same tools + same budgets, no planner/critic."""

from __future__ import annotations

import time
from pathlib import Path

from .agents.loop import parse_json_strict, trim_transcript
from .budgets import Budgets
from .llm import BaseProvider
from .schemas import Report
from .security import strip_injection_attempts, wrap_untrusted
from .tools.calculator import calculator
from .tools.fetch import fetch_page
from .tools.web_search import web_search

PROMPT = Path(__file__).resolve().parent.parent / "prompts" / "baseline.md"


def report_to_markdown(report: Report, sources: list[dict] | None = None) -> str:
    lines = [f"# {report.title}", "", report.summary, ""]
    for s in report.sections:
        lines += [f"## {s.heading}", "", s.body, ""]
    lines.append("## Sources")
    srcs = sources or [{"id": c} for c in report.citations]
    for sc in srcs:
        if isinstance(sc, dict):
            lines.append(f"- [{sc.get('id')}] {sc.get('title', '')} {sc.get('url', '')}")
        else:
            lines.append(f"- [{sc}]")
    return "\n".join(lines)


def run_baseline(
    question: str,
    llm: BaseProvider,
    model: str,
    budgets: Budgets | None = None,
    cache=None,
    trace=None,
    temperature: float = 0.0,
    seed: int | None = None,
    max_tokens: int = 1000,
    max_iters: int = 10,
    search_fn=None,
    fetch_fn=None,
    depth: str = "standard",
) -> dict:
    budgets = budgets or Budgets()
    budgets.start()
    if depth == "quick":
        max_iters = min(max_iters, 4)
    search_fn = search_fn or (lambda q: web_search(q, cache=cache, trace=trace))
    fetch_fn = fetch_fn or (lambda u: fetch_page(u, cache=cache, trace=trace))
    try:
        system = PROMPT.read_text(encoding="utf-8")
    except OSError:
        system = "Research and write cited report JSON."
    t0 = time.time()
    transcript = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"Research question: {question}\nUse TOOL: name | arg lines, then FINAL JSON.",
        },
    ]
    tool_calls = 0
    abort_reason = ""
    report: Report | None = None
    while True:
        reason = budgets.check()
        if reason:
            abort_reason = reason
            break
        transcript = trim_transcript(transcript)
        resp = llm.complete(transcript, model, temperature=temperature, seed=seed, max_tokens=max_tokens)
        if trace:
            trace.log_llm("baseline", model, resp.input_tokens, resp.output_tokens, resp.latency_s)
            try:
                trace.save()  # per-turn persistence: killed runs keep forensics
            except Exception:
                pass
        budgets.register_llm(resp.input_tokens + resp.output_tokens)
        txt = resp.text.strip()
        if txt.startswith("TOOL:"):
            _, rest = txt.split("TOOL:", 1)
            name, arg = [p.strip() for p in rest.split("|", 1)]
            budgets.register_tool()
            tool_calls += 1
            if name == "web_search":
                out = "\n".join(f"{r.get('title', '')} {r.get('url', '')}" for r in search_fn(arg)[:5])
            elif name == "fetch_page":
                pg = fetch_fn(arg)
                out = wrap_untrusted(strip_injection_attempts(pg.get("text", "")[:4000]))
            elif name == "calculator":
                out = calculator(arg)
            else:
                out = "ERROR: unknown tool"
            transcript += [
                {"role": "assistant", "content": txt},
                {"role": "user", "content": f"Tool result:\n{out}\nContinue or FINAL JSON."},
            ]
            if tool_calls >= max_iters:
                abort_reason = "baseline max tool iters reached"
                break
            continue
        if txt.startswith("{") or "FINAL" in txt:
            try:
                report = Report.model_validate(parse_json_strict(txt.replace("FINAL", "")))
                break
            except Exception as e:
                transcript.append({"role": "user", "content": f"Invalid JSON: {e}. Return valid Report JSON."})
                continue
        transcript += [
            {"role": "assistant", "content": txt},
            {"role": "user", "content": "Use TOOL: ... or FINAL JSON."},
        ]
    if report is None:
        report = Report(
            title=question[:80],
            summary=f"Partial result. Stopped: {abort_reason or 'no report'}.",
            sections=[],
            citations=[],
        )
    md = report_to_markdown(report)
    return {
        "report": report,
        "markdown": md,
        "abort_reason": abort_reason,
        "elapsed_s": round(time.time() - t0, 2),
        "llm_calls": budgets.llm_calls,
        "tool_calls": budgets.tool_calls,
        "tokens": budgets.tokens,
    }
