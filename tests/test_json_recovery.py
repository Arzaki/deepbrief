"""Recovery tests: malformed model JSON must trigger retry-with-feedback, not a run-killing crash."""

import json

import pytest

from deepbrief.agents.critic import run_critic
from deepbrief.agents.loop import parse_json_lenient
from deepbrief.agents.writer import run_writer
from deepbrief.llm import MockProvider
from deepbrief.schemas import Findings, Report

GOOD_REPORT = json.dumps(
    {
        "title": "T",
        "summary": "S",
        "sections": [{"heading": "H", "body": "body [S1]"}],
        "citations": ["S1"],
    }
)
GOOD_CRITIQUE = json.dumps({"issues": [], "verdict": "pass"})


def test_parse_json_lenient_allows_raw_newlines():
    bad_strict = '{"a": "line one\nline two"}'  # literal newline: strict json.loads rejects
    with pytest.raises(json.JSONDecodeError):
        json.loads(bad_strict)
    assert parse_json_lenient(bad_strict) == {"a": "line one\nline two"}


def test_parse_json_lenient_still_rejects_garbage():
    with pytest.raises(json.JSONDecodeError):
        parse_json_lenient("not json at all {{{")


def test_writer_recovers_after_one_malformed_reply():
    llm = MockProvider(script=['{"title": "T", "summary": broken json', GOOD_REPORT])
    rep = run_writer([Findings(sub_question_id="SQ1", claims=[], sources=[])], llm, "mock-agent")
    assert rep.title == "T"
    assert len(llm.calls) == 2  # initial + 1 feedback retry


def test_writer_raises_clear_error_after_retries_exhausted():
    llm = MockProvider(script=["garbage {{"] * 5)
    with pytest.raises(RuntimeError, match="writer failed to produce valid JSON"):
        run_writer([Findings(sub_question_id="SQ1", claims=[], sources=[])], llm, "mock-agent")
    assert len(llm.calls) == 3  # initial + 2 retries, then give up (no infinite loop)


def test_critic_recovers_after_malformed_reply():
    rep = Report(title="T", summary="S", sections=[], citations=[])
    llm = MockProvider(script=["{oops", GOOD_CRITIQUE])
    crit = run_critic(rep, [], llm, "mock-cheap")
    assert crit.verdict == "pass"
    assert len(llm.calls) == 2
