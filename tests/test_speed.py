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


def test_pipeline_researcher_timeout_continues_partial():
    """A hung researcher is abandoned after researcher_timeout_s; others still report."""
    import time as _time

    plan3 = json.dumps({"sub_questions": [{"id": f"SQ{i}", "text": f"q{i}"} for i in range(1, 4)]})
    empty = json.dumps({"sub_question_id": "SQ9", "claims": [], "sources": []})
    # Pop order across the 3 parallel researchers is nondeterministic, but the pool
    # contains exactly one hanging TOOL and two instant FINALs, so the outcome holds.
    llm = MockProvider(script=[plan3, "TOOL: fetch_page | http://slow.test/x", empty, empty])
    budgets = Budgets(max_llm_calls=50, max_tool_calls=50, max_wall_seconds=120)

    def slow_fetch(u):
        _time.sleep(5)
        return {"url": u, "title": "t", "text": "x"}

    out = run_pipeline(
        "q?",
        llm,
        "mock-agent",
        "mock-cheap",
        budgets=budgets,
        cache=_cache(),
        max_workers=2,
        researcher_timeout_s=2,
        use_critic=False,
        search_fn=lambda q: [],
        fetch_fn=slow_fetch,
    )
    assert len(out["findings"]) == 2  # the hung researcher was dropped, the other two returned
    assert "researcher_timeout" in out["abort_reason"]


def _calls_with(llm, needle):
    return sum(1 for c in llm.calls if any(needle in str(m.get("content", "")) for m in c["messages"]))


F1 = json.dumps(
    {
        "sub_question_id": "SQ1",
        "claims": [{"text": "c1", "source_id": "S1", "supporting_quote": "c1"}],
        "sources": [{"id": "S1", "url": "http://x.com/1", "title": "t"}],
    }
)
F2 = json.dumps(
    {
        "sub_question_id": "SQ2",
        "claims": [{"text": "c2", "source_id": "S2", "supporting_quote": "c2"}],
        "sources": [{"id": "S2", "url": "http://x.com/2", "title": "t"}],
    }
)
F3 = json.dumps({"sub_question_id": "SQ3", "claims": [], "sources": []})
HEAD = json.dumps({"title": "T", "summary": "S"})
SEC_A = json.dumps({"heading": "Part A", "body": "aaa [S1]"})
SEC_B = json.dumps({"heading": "Part B", "body": "bbb [S2]"})
SEC_B2 = json.dumps({"heading": "Part B", "body": "bbb fixed [S2]"})
REVISE_B = json.dumps(
    {"issues": [{"type": "unsupported_claim", "detail": "fix quote", "target": "Part B"}], "verdict": "revise"}
)
REVISE_VAGUE = json.dumps(
    {"issues": [{"type": "weak_source", "detail": "better sources", "target": ""}], "verdict": "revise"}
)


def test_pipeline_targeted_revision_regenerates_only_flagged_section():
    llm = MockProvider(script=[PLAN, F1, F2, F3, HEAD, SEC_A, SEC_B, REVISE_B, SEC_B2, PASS])
    out = run_pipeline(
        "q?",
        llm,
        "mock-agent",
        "mock-cheap",
        budgets=Budgets(max_tool_calls=50),
        cache=_cache(),
        max_workers=1,
        use_critic=True,
    )
    assert out["revision_rounds"] == 1
    assert [s.heading for s in out["report"].sections] == ["Part A", "Part B"]
    assert out["report"].sections[1].body == "bbb fixed [S2]"
    assert _calls_with(llm, "Write ONLY the report title") == 1  # head NOT rewritten
    assert _calls_with(llm, "ONE thorough report section") == 3  # 2 initial + 1 regen


def test_pipeline_revision_falls_back_on_unmapped_target():
    llm = MockProvider(script=[PLAN, F1, F2, F3, HEAD, SEC_A, SEC_B, REVISE_VAGUE, HEAD, SEC_A, SEC_B, PASS])
    out = run_pipeline(
        "q?",
        llm,
        "mock-agent",
        "mock-cheap",
        budgets=Budgets(max_tool_calls=50),
        cache=_cache(),
        max_workers=1,
        use_critic=True,
    )
    assert out["revision_rounds"] == 1
    assert _calls_with(llm, "Write ONLY the report title") == 2  # full rewrite fallback
    assert _calls_with(llm, "ONE thorough report section") == 4
