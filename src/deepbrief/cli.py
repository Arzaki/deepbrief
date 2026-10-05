"""CLI entry: run baseline or pipeline once, print markdown + trace path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from deepbrief.baseline import run_baseline  # noqa: E402
from deepbrief.budgets import Budgets  # noqa: E402
from deepbrief.cache import DiskCache  # noqa: E402
from deepbrief.config import Settings  # noqa: E402
from deepbrief.llm import make_provider  # noqa: E402
from deepbrief.pipeline import run_pipeline  # noqa: E402
from deepbrief.tracing import RunTrace  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--mode", default="multi", choices=["baseline", "multi", "multi-no-critic"])
    ap.add_argument(
        "--depth",
        default="standard",
        choices=["quick", "standard"],
        help="quick: fewer researcher turns, no critic (for surface-level questions)",
    )
    args = ap.parse_args()
    s = Settings()
    cache = DiskCache(s.cache_db, enabled=s.cache_enabled)
    trace = RunTrace(
        s.trace_dir,
        s.prompt_version,
        {"provider": s.llm_provider, "agent_model": s.resolved_agent_model(), "depth": args.depth},
    )
    llm = make_provider(s)
    budgets = Budgets(s.max_llm_calls, s.max_tool_calls, s.max_tokens, s.max_wall_seconds)
    if args.mode == "baseline":
        out = run_baseline(
            args.question,
            llm,
            s.resolved_agent_model(),
            budgets,
            cache,
            trace,
            s.temperature,
            s.seed,
            max_tokens=s.llm_max_tokens,
            depth=args.depth,
        )
    else:
        out = run_pipeline(
            args.question,
            llm,
            s.resolved_agent_model(),
            s.resolved_cheap_model(),
            budgets,
            cache,
            trace,
            s.temperature,
            s.seed,
            use_critic=(args.mode == "multi"),
            max_tokens=s.llm_max_tokens,
            depth=args.depth,
            max_workers=s.research_workers,
        )
    tp = trace.save()
    print(out["markdown"])
    print(f"\n[trace: {tp} | cost ${trace.totals['cost_usd']:.5f} | tokens {trace.totals['tokens']}]")


if __name__ == "__main__":
    main()
