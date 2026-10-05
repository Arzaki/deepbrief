"""Speed tests: parallel researchers, quick depth mode."""

import json

from deepbrief.baseline import run_baseline
from deepbrief.budgets import Budgets
from deepbrief.cache import DiskCache
from deepbrief.llm import MockProvider
from deepbrief.pipeline import run_pipeline

PLAN = json.dumps({"sub_questions": [{"id": f"SQ{i}", "text": f"q{i}"} for i in range(1, 4)]})
R_EMPTY = json.dumps({"sub_question_id": "SQ1", "claims": [], "sources": []})
WRITER = json.dumps({"title": "T", "summary": "S", "sections": [{"heading": "H", "body": "b"}], "citations": []})
PASS = json.dumps({"issues": [], "verdict": "pass"})


def _cache():
    return DiskCache(":memory:", enabled=False)


def test_pipeline_parallel_workers_complete():
    llm = MockProvider(script=[PLAN, R_EMPTY, R_EMPTY, R_EMPTY, WRITER, PASS])
    out = run_pipeline(
        "q?",
        llm,
        "mock-agent",
        "mock-cheap",
        budgets=Budgets(max_tool_calls=50),
        cache=_cache(),
        max_workers=3,
    )
    assert len(out["findings"]) == 3
    assert out["report"].title == "T"
    assert out["critique"].verdict == "pass"


def test_pipeline_quick_skips_critic_and_limits_turns():
    # Each researcher meanders (never FINAL) but is capped at 3 turns: 3 SQs × 3 = 9 calls.
    tools = [f"TOOL: web_search | speed {i}" for i in range(9)]
    stubs = dict(
        search_fn=lambda q: [],
        fetch_fn=lambda u: {"url": u, "title": "t", "text": "x"},
    )
    llm = MockProvider(script=[PLAN] + tools + [WRITER])
    out = run_pipeline(
        "q?",
        llm,
        "mock-agent",
        "mock-cheap",
        budgets=Budgets(max_tool_calls=50),
        cache=_cache(),
        depth="quick",
        max_workers=3,
        **stubs,
    )
    assert out["critique"] is None
    assert out["report"].title == "T"
    assert len(out["findings"]) == 3
    researcher_calls = [c for c in llm.calls if c["messages"][0]["content"].startswith("# Researcher")]
    assert len(researcher_calls) == 9


def test_pipeline_quick_sequential_matches_parallel():
    tools = [f"TOOL: web_search | speed {i}" for i in range(9)]
    stubs = dict(
        search_fn=lambda q: [],
        fetch_fn=lambda u: {"url": u, "title": "t", "text": "x"},
    )

    def _run(workers):
        return run_pipeline(
            "q?",
            MockProvider(script=[PLAN] + tools + [WRITER]),
            "mock-agent",
            "mock-cheap",
            budgets=Budgets(max_tool_calls=50),
            cache=_cache(),
            depth="quick",
            max_workers=workers,
            **stubs,
        )

    a, b = _run(1), _run(3)
    assert a["report"].title == b["report"].title == "T"
    assert len(a["findings"]) == len(b["findings"]) == 3


def test_baseline_quick_caps_tool_iters():
    llm = MockProvider(script=["TOOL: web_search | q"] * 20)
    out = run_baseline(
        "q?",
        llm,
        "mock-agent",
        budgets=Budgets(max_llm_calls=50, max_tool_calls=50),
        search_fn=lambda q: [{"title": "t", "url": "http://x", "snippet": "s"}],
        fetch_fn=lambda u: {"url": u, "title": "t", "text": "x"},
        depth="quick",
    )
    assert out["abort_reason"] == "baseline max tool iters reached"
    assert out["tool_calls"] == 4
