"""Multi-agent pipeline: Planner -> Researcher(s) -> Writer -> Critic (max 2 revisions)."""

from __future__ import annotations

import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .agents.critic import run_critic
from .agents.planner import run_planner
from .agents.researcher import run_researcher
from .agents.writer import run_writer
from .baseline import report_to_markdown
from .budgets import Budgets
from .llm import BaseProvider
from .verify import verify_findings


def run_pipeline(
    question: str,
    llm: BaseProvider,
    model: str,
    cheap_model: str | None = None,
    budgets: Budgets | None = None,
    cache=None,
    trace=None,
    temperature: float = 0.0,
    seed: int | None = None,
    max_tokens: int = 1000,
    use_critic: bool = True,
    max_revisions: int = 2,
    search_fn=None,
    fetch_fn=None,
    depth: str = "standard",
    max_workers: int = 3,
) -> dict:
    budgets = budgets or Budgets()
    budgets.start()
    cheap_model = cheap_model or model
    # Quick mode for surface-level questions: fewer researcher turns, no critic.
    # Same plan shape and budgets; just a cheaper gear for simple questions.
    quick = depth == "quick"
    r_iters = 3 if quick else 6
    if quick:
        use_critic = False
    t0 = time.time()
    abort_reason = ""
    plan = run_planner(
        question, llm, cheap_model, trace=trace, temperature=temperature, seed=seed, max_tokens=max_tokens
    )
    budgets.register_llm(50)  # planner accounting approximated via trace; keep counter moving
    all_findings = []
    page_texts: dict[str, str] = {}
    # One shared source-id namespace: per-researcher S1..Sn counters collide when
    # findings merge (observed: three different pages all listed as [S1]).
    _sid_counter = itertools.count(1)
    _sid_lock = threading.Lock()

    def _mint_sid() -> str:
        with _sid_lock:
            return f"S{next(_sid_counter)}"

    def _research(sq) -> tuple | None:
        if budgets.check():
            return None
        return run_researcher(
            sq.id,
            sq.text,
            llm,
            model,
            budgets=budgets,
            cache=cache,
            trace=trace,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            max_iters=r_iters,
            search_fn=search_fn,
            fetch_fn=fetch_fn,
            mint_sid=_mint_sid,
        )

    if max_workers <= 1:
        results = [_research(sq) for sq in plan.sub_questions]
    else:
        # Sub-questions are independent; budgets/cache/trace/LLM client are all thread-safe.
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_research, plan.sub_questions))
    for res in results:
        if res is None:
            abort_reason = budgets.check() or ""
            break
        f, pages = res
        all_findings.append(f)
        page_texts.update(pages)
    feedback = ""
    report = None
    critique = None
    rounds = 0
    while True:
        if budgets.check():
            abort_reason = budgets.check() or ""
            break
        report = run_writer(
            all_findings,
            llm,
            model,
            trace=trace,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            feedback=feedback,
        )
        budgets.register_llm(100)
        if not use_critic:
            break
        critique = run_critic(
            report,
            all_findings,
            llm,
            cheap_model,
            trace=trace,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
        )
        budgets.register_llm(50)
        if critique.verdict == "pass" or rounds >= max_revisions:
            if rounds >= max_revisions and critique.verdict == "revise":
                abort_reason = "revision_cap_reached(2)"
            break
        rounds += 1
        feedback = "\n".join(f"- [{i.type}] {i.detail}" for i in critique.issues)
    if report is None:
        from .schemas import Report as R

        report = R(
            title=question[:80],
            summary=f"Partial result. Stopped: {abort_reason}.",
            sections=[],
            citations=[],
        )
    flags = verify_findings(all_findings, page_texts)
    md = report_to_markdown(
        report,
        [{"id": s.id, "title": s.title, "url": s.url} for f in all_findings for s in f.sources],
    )
    return {
        "report": report,
        "markdown": md,
        "plan": plan,
        "findings": all_findings,
        "critique": critique,
        "revision_rounds": rounds,
        "citation_flags": flags,
        "abort_reason": abort_reason,
        "elapsed_s": round(time.time() - t0, 2),
    }
