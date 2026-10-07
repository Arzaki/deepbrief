"""Researcher: tool-calling loop for ONE sub-question -> Findings."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from ..llm import BaseProvider
from ..schemas import Findings, Source
from ..security import strip_injection_attempts, wrap_untrusted
from ..tools.calculator import calculator
from ..tools.fetch import fetch_page
from ..tools.web_search import web_search
from .loop import parse_json_lenient, retry_logger

PROMPT = Path(__file__).resolve().parents[3] / "prompts" / "researcher.md"

# Terminal pressure: the per-turn "Continue or FINAL" nudge lets models meander
# until turns run out (observed 2026-10-01: 4 researchers × 6 turns, zero FINALs).
# So the last turn forbids tools outright, and 2+ fetched pages flips the nudge
# from "continue" to "write it up now".
FINAL_TURN_DIRECTIVE = (
    "This is your LAST research turn. Do NOT call any more tools. "
    "Reply now with FINAL JSON Findings built from the evidence above "
    "(an empty claims list is acceptable if nothing was supported)."
)
EVIDENCE_SUFFICIENT_NUDGE = (
    "You have enough evidence (3+ fetched pages). "
    "Reply with FINAL JSON Findings now; only call another tool for a critical gap."
)


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
    max_searches: int = 3,
    search_fn=None,
    fetch_fn=None,
    mint_sid=None,
) -> tuple[Findings, dict[str, str]]:
    """Returns (findings, page_texts). search_fn/fetch_fn injectable for tests.

    mint_sid: optional shared () -> str ID minter. The pipeline passes one shared
    across researchers so source ids are globally unique; without it each call
    mints its own S1..Sn namespace (fine standalone, colliding when merged).
    """
    search_fn = search_fn or (lambda q: web_search(q, cache=cache, trace=trace))
    fetch_fn = fetch_fn or (lambda u: fetch_page(u, cache=cache, trace=trace))
    try:
        system = PROMPT.read_text(encoding="utf-8")
    except OSError:
        system = "Answer with tools. Return strict JSON Findings."
    context_blocks: list[str] = []
    sources: list[Source] = []
    page_texts: dict[str, str] = {}
    _local_sid = 0

    def _save_trace() -> None:
        # Persist every turn: a killed run still leaves the exact hanging call
        # visible (last logged event with no successor). Cheap, never fatal.
        saver = getattr(trace, "save", None)
        if callable(saver):
            try:
                saver()
            except Exception:
                pass

    def _mint() -> str:
        nonlocal _local_sid
        if mint_sid is not None:
            return mint_sid()
        _local_sid += 1
        return f"S{_local_sid}"

    searches_used = 0
    seen_queries: set[str] = set()
    saw_results = False
    fetch_forced_once = False

    def _tool(name: str, arg: str) -> str:
        nonlocal searches_used, saw_results
        if budgets:
            budgets.register_tool()
            if trace:
                pass
        if name == "web_search":
            # Anti-flail rules (deterministic, no LLM needed): cap searches per
            # sub-question and never repeat a query. Observed failure 2026-10-01:
            # 20 near-identical searches for a typo ("biplor") burned the run budget.
            q = arg.strip().lower()
            if q in seen_queries:
                return "Already searched this exact query. Do NOT search again — fetch a URL or reply FINAL JSON."
            if searches_used >= max_searches:
                return (
                    f"Search budget used ({max_searches} per sub-question). "
                    "Fetch a promising URL from the results above or reply FINAL JSON now."
                )
            seen_queries.add(q)
            searches_used += 1
            res = search_fn(arg)
            if res:
                saw_results = True
            lines = []
            for i, r in enumerate(res[:5], start=1):
                lines.append(f"[{i}]: {r.get('title', '')} {r.get('url', '')} — {r.get('snippet', '')[:200]}")
            if not lines:
                return "0 results. Reformulate ONCE (check for typos/alternate terms), or fetch a known URL, or FINAL."
            return "Search results (fetch by URL to get a citable S-id):\n" + "\n".join(lines)
        if name in ("fetch_page", "fetch_pages"):
            # Batch form fetches up to 3 URLs concurrently in ONE turn, so depth
            # doesn't cost extra researcher turns (fetches are free, only time).
            # URLs are whitespace-separated (trailing commas tolerated); embedded
            # commas inside a URL are preserved.
            toks = [t.strip().strip(",;") for t in arg.split()]
            urls = [t for t in toks if t.startswith("http")]
            urls = urls[:1] if name == "fetch_page" else urls[:3]
            if not urls:
                return "ERROR: no valid http(s) URL found. Provide full URLs to fetch."
            with ThreadPoolExecutor(max_workers=3) as pool:
                fetched = list(pool.map(fetch_fn, urls))
            blocks = []
            for url, pg in zip(urls, fetched, strict=True):
                pg = pg or {}
                err = pg.get("error", "")
                text = strip_injection_attempts((pg.get("text", "") or "")[:4000])
                sid = _mint()
                sources.append(
                    Source(
                        id=sid,
                        url=url,
                        title=pg.get("title", ""),
                        fetched_at=datetime.now(UTC).isoformat(),
                    )
                )
                page_texts[sid] = text
                if text:
                    blocks.append(
                        f"Fetched [{sid}] {url} — cite this as [{sid}] with a verbatim quote:\n"
                        + wrap_untrusted(text[:4000])
                    )
                else:
                    blocks.append(f"Fetched [{sid}] {url} — EMPTY or failed ({err[:150]}). Try another URL.")
            return "\n\n".join(blocks)
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
    for turn in range(max_iters):
        if budgets and budgets.check():
            break
        if turn == max_iters - 1:
            transcript.append({"role": "user", "content": FINAL_TURN_DIRECTIVE})
        resp = llm.complete(
            transcript,
            model,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            on_retry=retry_logger(trace, "researcher"),
        )
        if trace:
            trace.log_llm(
                "researcher",
                model,
                resp.input_tokens,
                resp.output_tokens,
                resp.latency_s,
                sub_question=sub_qid,
                excerpt=resp.text[:200],
            )
        if budgets:
            budgets.register_llm(resp.input_tokens + resp.output_tokens)
        _save_trace()
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
            if len(sources) >= 3:
                transcript.append({"role": "user", "content": f"Tool result:\n{out}\n{EVIDENCE_SUFFICIENT_NUDGE}"})
            else:
                transcript.append({"role": "user", "content": f"Tool result:\n{out}\nContinue or give FINAL JSON."})
            continue
        if "FINAL" in txt or txt.startswith("{"):
            try:
                data = parse_json_lenient(txt.replace("FINAL", "").strip())
                data.setdefault("sub_question_id", sub_qid)
                f = Findings.model_validate(data)
                # No-fetch FINAL with un-fetched results on the table: send the model
                # back once to fetch (observed 2026-10-01 twice: searches without a
                # single fetch → empty findings → zero-coverage report; the second
                # time the model even listed fabricated sources, so only ACTUAL
                # fetches count here, not model-declared source entries).
                if not sources and saw_results and not fetch_forced_once:
                    fetch_forced_once = True
                    transcript.append(
                        {
                            "role": "user",
                            "content": "You have search results you never fetched. Fetch the most "
                            "promising URL now (fetch_pages takes up to 3) — FINAL with zero "
                            "fetched pages is not accepted on the first attempt.",
                        }
                    )
                    continue
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
        if len(sources) >= 3:
            transcript.append({"role": "user", "content": EVIDENCE_SUFFICIENT_NUDGE})
        else:
            transcript.append({"role": "user", "content": "Reply with TOOL: ... or FINAL JSON Findings."})
    # Graceful degradation: return what we have
    return Findings(sub_question_id=sub_qid, claims=[], sources=sources), page_texts
