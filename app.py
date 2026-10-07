"""Gradio UI: question box, live progress, cited report, trace view, cost/latency."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr

from deepbrief.baseline import run_baseline
from deepbrief.budgets import Budgets
from deepbrief.cache import DiskCache
from deepbrief.config import Settings
from deepbrief.llm import make_provider
from deepbrief.pipeline import run_pipeline
from deepbrief.tracing import RunTrace

settings = Settings()
SESSION_RUNS: dict[str, int] = {}
SESSION_CAP = 5


def linkify_citations(md: str, url_map: dict[str, str]) -> str:
    def _repl(m):
        sid = m.group(1)
        url = url_map.get(sid)
        if url:
            return f"[[{sid}]({url})]"
        return m.group(0)

    return re.sub(r"\[(\w\d+)\]", _repl, md)


def run_fn(question: str, mode: str, depth: str, session: str, progress: gr.Progress | None = None):
    progress = progress or gr.Progress()
    if not question.strip():
        return "Please enter a question.", "", ""
    key = session or "default"
    SESSION_RUNS[key] = SESSION_RUNS.get(key, 0) + 1
    if SESSION_RUNS[key] > SESSION_CAP:
        return "Session run cap reached (5). Please restart the demo.", "", ""
    s = Settings()
    cache = DiskCache(s.cache_db, enabled=s.cache_enabled)
    trace = RunTrace(
        s.trace_dir,
        s.prompt_version,
        {"provider": s.llm_provider, "agent_model": s.resolved_agent_model(), "mode": mode, "depth": depth},
    )
    llm = make_provider(s)
    budgets = Budgets(s.max_llm_calls, s.max_tool_calls, s.max_tokens, s.max_wall_seconds)
    progress(0.1, desc="Planning...")
    t0 = time.time()
    try:
        if mode == "baseline":
            out = run_baseline(
                question,
                llm,
                s.resolved_agent_model(),
                budgets,
                cache,
                trace,
                s.temperature,
                s.seed,
                max_tokens=s.llm_max_tokens,
                depth=depth,
            )
            url_map = {}
        else:
            progress(0.3, desc="Researching sub-questions...")
            out = run_pipeline(
                question,
                llm,
                s.resolved_agent_model(),
                s.resolved_cheap_model(),
                budgets,
                cache,
                trace,
                s.temperature,
                s.seed,
                use_critic=(mode == "multi-critic"),
                max_tokens=s.llm_max_tokens,
                depth=depth,
                max_workers=s.research_workers,
                researcher_timeout_s=s.researcher_timeout_s,
            )
            url_map = {src.id: src.url for f in out.get("findings", []) for src in f.sources}
            progress(0.9, desc="Writing + critiquing...")
    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        msg = str(e)
        hint = ""
        if "model_not_found" in msg or "does not exist" in msg:
            hint = (
                "\n\n**The configured model ID is retired or unavailable on your plan.** "
                "Run `uv run python scripts/check_models.py` to list live IDs, then set "
                "`AGENT_MODEL` / `CHEAP_MODEL` in `.env` to a listed one."
            )
        tp = trace.save()  # partial trace: diagnosis survives even failed runs
        stats = (
            f"FAILED after {elapsed}s | llm {trace.totals['llm_calls']} | "
            f"tools {trace.totals['tool_calls']} | tokens {trace.totals['tokens']}"
        )
        return f"Run failed after {elapsed}s: {msg}{hint}", f"{stats}\n\nPartial trace: {tp}", ""
    tp = trace.save()
    elapsed = round(time.time() - t0, 1)
    stats = (
        f"cost ≈ ${trace.totals['cost_usd']:.5f} | tokens {trace.totals['tokens']} | "
        f"latency {elapsed}s | llm {trace.totals['llm_calls']} | tools {trace.totals['tool_calls']} | "
        f"abort: {out.get('abort_reason') or 'none'}"
    )
    md = linkify_citations(out["markdown"], url_map)
    trace_json = tp.read_text()[:20000]
    return md, f"{stats}\n\nTrace: {tp}", trace_json


with gr.Blocks(title="DeepBrief") as demo:
    gr.Markdown("# DeepBrief — multi-agent research reports (cited)")
    q = gr.Textbox(
        label="Research question",
        placeholder="Compare vector databases for a startup with ~1M documents",
    )
    mode = gr.Radio(["baseline", "multi-no-critic", "multi-critic"], value="multi-critic", label="Mode")
    depth = gr.Radio(
        ["quick", "standard"],
        value="quick",
        label="Depth (quick: fewer turns, no critic — for surface-level questions)",
    )
    session = gr.Textbox(label="Session id (rate-limit key)", value="demo")
    btn = gr.Button("Run research")
    report = gr.Markdown(label="Report")
    stats = gr.Textbox(label="Cost / latency")
    trace_view = gr.Code(label="Trace (JSON)", language="json")
    btn.click(run_fn, inputs=[q, mode, depth, session], outputs=[report, stats, trace_view])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
