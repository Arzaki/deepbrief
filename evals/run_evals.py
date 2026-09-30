"""Run evals across configs: baseline | multi-no-critic | multi-critic. Offline-friendly."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

from deepbrief.baseline import run_baseline  # noqa: E402
from deepbrief.budgets import Budgets  # noqa: E402
from deepbrief.cache import DiskCache  # noqa: E402
from deepbrief.config import Settings  # noqa: E402
from deepbrief.llm import MockProvider, make_judge_provider, make_provider  # noqa: E402
from deepbrief.pipeline import run_pipeline  # noqa: E402
from deepbrief.tracing import RunTrace  # noqa: E402
from evals.judges import judge_coverage, judge_entailment  # noqa: E402
from evals.metrics import coverage_score, entailment_scores  # noqa: E402


def load_questions(path: Path, limit: int | None = None, offline: bool = False):
    qs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if offline:
        qs = qs[:3]
    if limit:
        qs = qs[:limit]
    return qs


def run_one(question: dict, config: str, settings: Settings, cache, judge, judge_model: str) -> dict:
    budgets = Budgets(
        max_llm_calls=settings.max_llm_calls,
        max_tool_calls=settings.max_tool_calls,
        max_tokens=settings.max_tokens,
        max_wall_seconds=settings.max_wall_seconds,
    )
    trace = RunTrace(
        settings.trace_dir,
        settings.prompt_version,
        {
            "provider": settings.llm_provider,
            "agent_model": settings.resolved_agent_model(),
            "config": config,
            "temperature": settings.temperature,
        },
    )
    llm = make_provider(settings)
    t0 = time.time()
    # Offline/mock runs must not touch the network: stub the tools.
    stubs: dict = {}
    if llm.name == "mock":
        stubs = {
            "search_fn": lambda q: [{"title": "mock result", "url": "http://example.com", "snippet": "mock snippet"}],
            "fetch_fn": lambda u: {"url": u, "title": "mock page", "text": "mock page text"},
        }
    if config == "baseline":
        out = run_baseline(
            question["question"],
            llm,
            settings.resolved_agent_model(),
            budgets=budgets,
            cache=cache,
            trace=trace,
            temperature=settings.temperature,
            seed=settings.seed,
            **stubs,
        )
        findings_claims: list[dict] = []
    elif config == "multi-no-critic":
        out = run_pipeline(
            question["question"],
            llm,
            settings.resolved_agent_model(),
            settings.resolved_cheap_model(),
            budgets=budgets,
            cache=cache,
            trace=trace,
            temperature=settings.temperature,
            seed=settings.seed,
            use_critic=False,
            max_revisions=settings.max_revisions,
            **stubs,
        )
        findings_claims = [
            {"text": c.text, "supporting_quote": c.supporting_quote, "page_text": ""}
            for f in out["findings"]
            for c in f.claims
        ]
    else:
        out = run_pipeline(
            question["question"],
            llm,
            settings.resolved_agent_model(),
            settings.resolved_cheap_model(),
            budgets=budgets,
            cache=cache,
            trace=trace,
            temperature=settings.temperature,
            seed=settings.seed,
            use_critic=True,
            max_revisions=settings.max_revisions,
            **stubs,
        )
        findings_claims = [
            {"text": c.text, "supporting_quote": c.supporting_quote, "page_text": ""}
            for f in out["findings"]
            for c in f.claims
        ]
    md = out["markdown"]
    cov = judge_coverage(md, question.get("must_have_facts", []), judge, judge_model)
    ent = judge_entailment(
        findings_claims
        or [{"text": s.heading, "supporting_quote": "", "page_text": md} for s in out["report"].sections],
        judge,
        judge_model,
    )
    latency = round(time.time() - t0, 2)
    trace_path = str(trace.save())
    return {
        "id": question["id"],
        "config": config,
        "coverage": coverage_score(cov.judgments),
        **entailment_scores(ent.judgments),
        "latency_s": latency,
        "tokens": trace.totals["tokens"],
        "cost_usd": round(trace.totals["cost_usd"], 6),
        "llm_calls": trace.totals["llm_calls"],
        "tool_calls": trace.totals["tool_calls"],
        "trace": trace_path,
        "report": md[:2000],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="baseline,multi-no-critic,multi-critic")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--questions", default=str(ROOT / "questions.jsonl"))
    args = ap.parse_args()
    settings = Settings()
    if args.offline:
        settings.llm_provider = "mock"
        settings.judge_provider = "mock"
    cache = DiskCache(settings.cache_db, enabled=settings.cache_enabled)
    judge = make_judge_provider(settings) if not args.offline else MockProvider()
    judge_model = settings.resolved_judge_model() if not args.offline else "mock"
    questions = load_questions(Path(args.questions), args.limit, args.offline)
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    rows: list[dict] = []
    for cfg in args.configs.split(","):
        for q in questions:
            rows.append(run_one(q, cfg.strip(), settings, cache, judge, judge_model))
    # per-question CSV
    csv_path = RESULTS / f"results_{stamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "report"])
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if k != "report"})
    # summary table
    summary = summarize(rows)
    config_pin = {
        "models": {
            "agent": settings.resolved_agent_model(),
            "cheap": settings.resolved_cheap_model(),
            "judge": judge_model,
        },
        "temperature": settings.temperature,
        "prompt_version": settings.prompt_version,
        "date": stamp,
        "cache": settings.cache_enabled,
        "offline": args.offline,
    }
    md_path = RESULTS / f"SUMMARY_{stamp}.md"
    md_path.write_text(
        "# Eval results\n\n"
        + f"Config: `{json.dumps(config_pin)}`\n\n"
        + summary
        + f"\n\nPer-question CSV: `{csv_path.name}`\n",
        encoding="utf-8",
    )
    # spot-check CSV: 10 random claims
    spot = RESULTS / f"spotcheck_{stamp}.csv"
    sample = random.sample(rows, min(10, len(rows)))
    with open(spot, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "config", "report_excerpt"])
        w.writeheader()
        for r in sample:
            w.writerow({"id": r["id"], "config": r["config"], "report_excerpt": r["report"][:500]})
    print(f"Wrote {csv_path}\n{md_path}\n{spot}\n{summary}")


def summarize(rows: list[dict]) -> str:
    from collections import defaultdict

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["config"]].append(r)
    lines = [
        "| config | n | coverage | citation_acc | faithfulness | avg_latency_s | avg_cost_usd |",
        "|---|---|---|---|---|---|---|",
    ]
    for cfg, rs in groups.items():
        n = len(rs)
        cov = sum(r["coverage"] for r in rs) / n
        ca = sum(r["citation_accuracy"] for r in rs) / n
        fa = sum(r["faithfulness"] for r in rs) / n
        lat = sum(r["latency_s"] for r in rs) / n
        cost = sum(r["cost_usd"] for r in rs) / n
        lines.append(f"| {cfg} | {n} | {cov:.3f} | {ca:.3f} | {fa:.3f} | {lat:.1f} | {cost:.5f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
