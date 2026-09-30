"""M1 unit tests: config, retry/backoff, cache, tracing, costs, schemas."""

import json

import pytest

from deepbrief.cache import DiskCache
from deepbrief.config import Settings, get_price
from deepbrief.costs import estimate_cost
from deepbrief.llm import LLMError, MockProvider, compute_backoff, with_retries
from deepbrief.schemas import Plan
from deepbrief.tracing import RunTrace


def test_settings_defaults_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    s = Settings()
    assert s.resolved_agent_model() == "mock-agent"


def test_backoff_respects_retry_after():
    w = compute_backoff(0, base=1.0, retry_after=25.0)
    assert w >= 25.0
    w2 = compute_backoff(3, base=1.0)
    assert w2 > 0


def test_with_retries_eventual_success():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise LLMError("rate limited", status=429)
        return "ok"

    assert with_retries(fn, max_retries=4, base=0.001) == "ok"
    assert calls["n"] == 3


def test_with_retries_non_retryable():
    def fn():
        raise LLMError("bad request", status=400)

    with pytest.raises(LLMError):
        with_retries(fn, max_retries=2, base=0.001)


def test_cache_put_get(tmp_path):
    c = DiskCache(str(tmp_path / "c.sqlite"), enabled=True)
    assert c.get("search", "q") is None
    c.put("search", "q", "hello")
    assert c.get("search", "q") == "hello"
    assert c.hits == 1 and c.misses == 1
    c.close()


def test_cache_disabled():
    c = DiskCache(":memory:", enabled=False)
    c.put("k", "q", "v")
    assert c.get("k", "q") is None


def test_trace_logs_and_saves(tmp_path):
    t = RunTrace(str(tmp_path), "v1", {"llm_provider": "mock"})
    t.log_llm("planner", "mock-agent", 10, 5, 0.01)
    t.log("tool", "web_search", cache_hit=True)
    p = t.save()
    data = json.loads(p.read_text())
    assert data["totals"]["llm_calls"] == 1
    assert data["totals"]["tool_calls"] == 1
    assert data["prompt_version"] == "v1"


def test_cost_estimator():
    c = estimate_cost("mock-agent", 1000, 1000)
    assert c == 0.0
    c2 = estimate_cost("llama-3.3-70b-versatile", 1000, 1000)
    assert c2 > 0
    assert get_price("nope")["input"] >= 0


def test_plan_validation_rejects_two_items():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Plan.model_validate({"sub_questions": [{"id": "SQ1", "text": "a"}, {"id": "SQ2", "text": "b"}]})


def test_plan_validation_ok():
    p = Plan.model_validate({"sub_questions": [{"id": f"SQ{i}", "text": f"q{i}"} for i in range(1, 4)]})
    assert len(p.sub_questions) == 3


def test_mock_provider():
    m = MockProvider(script=['{"verdict": "pass", "issues": []}'])
    r = m.complete([{"role": "user", "content": "hi"}], "mock-agent")
    assert "pass" in r.text
