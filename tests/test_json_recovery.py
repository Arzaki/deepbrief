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


def test_writer_falls_back_after_retries_exhausted():
    """Spec change: persistent model failure now yields a deterministic cited
    fallback (faithfulness holds — only researcher claims listed), not a crash."""
    from deepbrief.agents.writer import _run_writer_llm

    llm = MockProvider(script=["garbage {{"] * 5)
    findings = [Findings(sub_question_id="SQ1", claims=[], sources=[])]
    with pytest.raises(RuntimeError, match="writer failed to produce valid JSON"):
        _run_writer_llm(findings, llm, "mock-agent")
    assert len(llm.calls) == 3  # initial + 2 retries, then give up (no infinite loop)

    llm2 = MockProvider(script=["garbage {{"] * 5)
    rep = run_writer(findings, llm2, "mock-agent")
    assert "without LLM" in rep.title


def test_critic_recovers_after_malformed_reply():
    rep = Report(title="T", summary="S", sections=[], citations=[])
    llm = MockProvider(script=["{oops", GOOD_CRITIQUE])
    crit = run_critic(rep, [], llm, "mock-cheap")
    assert crit.verdict == "pass"
    assert len(llm.calls) == 2


def test_writer_falls_back_on_llm_error_with_claims_intact():
    from deepbrief.llm import LLMError
    from deepbrief.schemas import Claim, Source

    findings = [
        Findings(
            sub_question_id="SQ1",
            claims=[Claim(text="LDAC hits 990 kbps", source_id="S1", supporting_quote="990 kbps")],
            sources=[Source(id="S1", url="http://x.com/1", title="t")],
        )
    ]
    llm = MockProvider(script=[LLMError("Rate limited", status=429)])
    rep = run_writer(findings, llm, "mock-agent")
    assert "without LLM" in rep.title
    assert "LDAC hits 990 kbps [S1]" in rep.sections[0].body
    assert rep.citations == ["S1"]


def test_critic_falls_back_to_pass_on_llm_error():
    from deepbrief.llm import LLMError

    rep = Report(title="T", summary="S", sections=[], citations=[])
    llm = MockProvider(script=[LLMError("boom", status=500)])
    crit = run_critic(rep, [], llm, "mock-cheap")
    assert crit.verdict == "pass" and crit.issues == []
