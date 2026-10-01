"""Researcher: tool-calling loop for ONE sub-question -> Findings."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..llm import BaseProvider
from ..schemas import Findings, Source
from ..security import strip_injection_attempts, wrap_untrusted
from ..tools.calculator import calculator
from ..tools.fetch import fetch_page
from ..tools.web_search import web_search
from .loop import parse_json_strict

PROMPT = Path(__file__).resolve().parents[3] / "prompts" / "researcher.md"


def run_researcher(
    sub_qid: str,
    sub_qtext: str,
    llm: BaseProvider,
    model: str,
    budgets=None,
    cache=None,
    trace=None,
    temperature: float = 0.0,
    seed: int | None = None,
    max_tokens: int = 1000,
    max_iters: int = 6,
    search_fn=None,
    fetch_fn=None,
) -> tuple[Findings, dict[str, str]]:
    """Returns (findings, page_texts). search_fn/fetch_fn injectable for tests."""
    search_fn = search_fn or (lambda q: web_search(q, cache=cache, trace=trace))
    fetch_fn = fetch_fn or (lambda u: fetch_page(u, cache=cache, trace=trace))
    try:
        system = PROMPT.read_text(encoding="utf-8")
    except OSError:
        system = "Answer with tools. Return strict JSON Findings."
    context_blocks: list[str] = []
    sources: list[Source] = []
    page_texts: dict[str, str] = {}
    sid_counter = 0

    def _tool(name: str, arg: str) -> str:
        nonlocal sid_counter
        if budgets:
            budgets.register_tool()
            if trace:
                pass
        if name == "web_search":
            res = search_fn(arg)
            lines = []
            for r in res[:5]:
                sid_counter += 1
                sid = f"S{sid_counter}"
                lines.append(f"{sid}: {r.get('title', '')} {r.get('url', '')} — {r.get('snippet', '')[:200]}")
            return "\n".join(lines) or "no results"
        if name == "fetch_page":
            pg = fetch_fn(arg)
            text = strip_injection_attempts(pg.get("text", "")[:4000])
            sid_counter += 1
            sid = f"S{sid_counter}"
            sources.append(
                Source(
                    id=sid,
                    url=arg,
                    title=pg.get("title", ""),
                    fetched_at=datetime.now(UTC).isoformat(),
                )
            )
            page_texts[sid] = text
            return wrap_untrusted(text[:4000])
        if name == "calculator":
            return calculator(arg)
        return "ERROR: unknown tool"

    transcript = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"Sub-question [{sub_qid}]: {sub_qtext}\n"
                "Use tools by replying with TOOL: name | arg. "
                "When done, reply with FINAL JSON Findings."
            ),
        },
    ]
    for _ in range(max_iters):
        if budgets and budgets.check():
            break
        resp = llm.complete(transcript, model, temperature=temperature, seed=seed, max_tokens=max_tokens)
        if trace:
            trace.log_llm(
                "researcher",
                model,
                resp.input_tokens,
                resp.output_tokens,
                resp.latency_s,
                sub_question=sub_qid,
            )
        if budgets:
            budgets.register_llm(resp.input_tokens + resp.output_tokens)
        txt = resp.text.strip()
        if txt.startswith("TOOL:"):
            try:
                _, rest = txt.split("TOOL:", 1)
                name, arg = [p.strip() for p in rest.split("|", 1)]
            except ValueError:
                transcript.append({"role": "user", "content": "Bad TOOL format. Use TOOL: name | arg"})
                continue
            out = _tool(name, arg)
            context_blocks.append(out)
            transcript.append({"role": "assistant", "content": txt})
            transcript.append({"role": "user", "content": f"Tool result:\n{out}\nContinue or give FINAL JSON."})
            continue
        if "FINAL" in txt or txt.startswith("{"):
            try:
                data = parse_json_strict(txt.replace("FINAL", "").strip())
                data.setdefault("sub_question_id", sub_qid)
                f = Findings.model_validate(data)
                # attach discovered sources not already listed
                known = {s.id for s in f.sources}
                for s in sources:
                    if s.id not in known:
                        f.sources.append(s)
                return f, page_texts
            except Exception as e:
                transcript.append(
                    {
                        "role": "user",
                        "content": f"JSON invalid: {e}. Return ONLY valid Findings JSON.",
                    }
                )
                continue
        transcript.append({"role": "assistant", "content": txt})
        transcript.append({"role": "user", "content": "Reply with TOOL: ... or FINAL JSON Findings."})
    # Graceful degradation: return what we have
    return Findings(sub_question_id=sub_qid, claims=[], sources=sources), page_texts
