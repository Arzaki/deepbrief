"""Multi-agent pipeline: Planner -> Researcher(s) -> Writer -> Critic (max 2 revisions)."""

from __future__ import annotations

import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

from .agents.critic import run_critic
from .agents.planner import run_planner
from .agents.researcher import run_researcher
from .agents.writer import run_sections, run_writer
from .baseline import report_to_markdown
from .budgets import Budgets
from .llm import BaseProvider
from .schemas import Report
from .verify import verify_findings


def _section_index_for_issue(target: str | None, sections) -> int | None:
    """Map a critic issue onto a section index via its heading; None if unmappable."""
    t = (target or "").strip().lower()
    if not t:
        return None
    for i, s in enumerate(sections):
        h = (s.heading or "").strip().lower()
        if h and (h in t or t in h):
            return i
    return None


def _revise_report(
    report: Report,
    findings,
    critique,
    llm,
    model,
    trace=None,
    temperature: float = 0.0,
    seed: int | None = None,
    max_tokens: int = 1000,
    max_workers: int = 3,
) -> Report:
    """Regenerate only the critic-flagged sections; full rewrite if issues don't map."""
    groups = [f for f in findings if f.claims]
    per_group: dict[str, str] = {}
    idxs: list[int] = []
    mappable = critique is not None and bool(critique.issues) and len(report.sections) == len(groups)
    if mappable:
        for issue in critique.issues:
            j = _section_index_for_issue(issue.target, report.sections)
            if j is None:  # an issue we can't place: don't silently drop it
                mappable = False
                break
            if j not in idxs:
                idxs.append(j)
            key = groups[j].sub_question_id
            per_group[key] = per_group.get(key, "") + f"- [{issue.type}] {issue.detail}\n"
    if not mappable or not idxs:
        feedback = "\n".join(f"- [{i.type}] {i.detail}" for i in critique.issues) if critique else ""
        return run_writer(
            findings,
            llm,
            model,
            trace=trace,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            feedback=feedback,
            max_workers=max_workers,
        )
    subset = [groups[j] for j in idxs]
    fresh = run_sections(
        subset,
        llm,
        model,
        trace=trace,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
        max_workers=max_workers,
        per_group_feedback=per_group,
    )
    sections = list(report.sections)
    for j, sec in zip(idxs, fresh, strict=True):
        sections[j] = sec
    return Report(title=report.title, summary=report.summary, sections=sections, citations=report.citations)


def _checkpoint(trace, stage: str) -> None:
    """Re-save the growing trace at stage boundaries.

    trace.save() always writes the same run_<id>.json, so this is cheap. A run
    killed mid-flight (timeout, Ctrl+C) still leaves everything up to the last
    completed stage for diagnosis — previously such runs left nothing at all.
    """
    if trace is None:
        return
    try:
        trace.save()
    except Exception:
        pass  # checkpointing must never break the run itself


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
    researcher_timeout_s: int = 240,
    verbose: bool = False,
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
    if verbose:
        print(f"[plan] {len(plan.sub_questions)} sub-questions", flush=True)
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
        # Each future is bounded: a hung researcher is abandoned (partial) instead of
        # stalling the run past every other timeout (observed 2026-10-01: 600s, zero findings).
        # NOTE: explicit shutdown(wait=False) — a `with` block would join hung workers and
        # defeat the timeouts. A truly hung native call may leak its worker thread.
        pool = ThreadPoolExecutor(max_workers=max_workers)
        try:
            futs = [pool.submit(_research, sq) for sq in plan.sub_questions]
            results = []
            for sq, fut in zip(plan.sub_questions, futs, strict=True):
                try:
                    results.append(fut.result(timeout=researcher_timeout_s))
                except FuturesTimeout:
                    results.append(None)
                    note = f"researcher_timeout({sq.id}>{researcher_timeout_s}s)"
                    ab = f"{abort_reason}; " if abort_reason else ""
                    abort_reason = ab + note
                    if verbose:
                        print(f"[research timeout] {note}; continuing partial", flush=True)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
    partial = False
    for res in results:
        if res is None:
            partial = True
            continue
        f, pages = res
        all_findings.append(f)
        page_texts.update(pages)
    if partial and not abort_reason:
        abort_reason = budgets.check() or "partial_researcher_results"
    if verbose:
        n_claims = sum(len(f.claims) for f in all_findings)
        print(
            f"[research done] {len(all_findings)} findings, {n_claims} claims. abort={abort_reason or 'none'}",
            flush=True,
        )
    _checkpoint(trace, "research done")
    report = None
    critique = None
    rounds = 0
    while True:
        if budgets.check():
            abort_reason = budgets.check() or ""
            break
        if report is None:
            report = run_writer(
                all_findings,
                llm,
                model,
                trace=trace,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
                feedback="",
                max_workers=max_workers,
            )
        else:
            # Targeted revision: regenerate ONLY the sections the critic flagged
            # (a full rewrite costs head + every section; measured 2026-10-01 a
            # revision round at ~130s of queued calls). Falls back to a full
            # rewrite when issues don't map cleanly onto sections.
            report = _revise_report(
                report,
                all_findings,
                critique,
                llm,
                model,
                trace=trace,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
                max_workers=max_workers,
            )
        budgets.register_llm(100)
        if verbose:
            print(f"[writer] round {rounds}: {len(report.sections)} sections", flush=True)
        _checkpoint(trace, f"writer round {rounds}")
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
        if verbose:
            print(f"[critic] verdict={critique.verdict} issues={len(critique.issues)}", flush=True)
        if critique.verdict == "pass" or rounds >= max_revisions:
            if rounds >= max_revisions and critique.verdict == "revise":
                abort_reason = "revision_cap_reached(2)"
            break
        rounds += 1
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
