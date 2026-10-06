"""Depth tests: section-wise writer breaks the per-call output ceiling."""

import json

from deepbrief.agents.writer import run_writer
from deepbrief.llm import MockProvider
from deepbrief.schemas import Claim, Findings, Source


def _findings(sqid, claims):
    return Findings(
        sub_question_id=sqid,
        claims=[Claim(text=t, source_id=sid, supporting_quote=q) for t, sid, q in claims],
        sources=[Source(id=sid, url=f"http://x.com/{sid}", title="t") for _, sid, _ in claims],
    )


HEAD = json.dumps({"title": "Big Report", "summary": "Covers A and B."})
SEC_A = json.dumps({"heading": "Part A", "body": "Long thorough section about A [S1] [S2]."})
SEC_B = json.dumps({"heading": "Part B", "body": "Long thorough section about B [S3]."})
SEC_SAME = json.dumps({"heading": "H", "body": "Body text here [S1]."})


def test_writer_section_path_sequential():
    f1 = _findings("SQ1", [("claim one", "S1", "quote one"), ("claim two", "S2", "quote two")])
    f2 = _findings("SQ2", [("claim three", "S3", "quote three")])
    llm = MockProvider(script=[HEAD, SEC_A, SEC_B])
    rep = run_writer([f1, f2], llm, "mock-agent", max_workers=1)
    assert rep.title == "Big Report"
    assert rep.summary == "Covers A and B."
    assert [s.heading for s in rep.sections] == ["Part A", "Part B"]
    assert rep.citations == ["S1", "S2", "S3"]
    assert len(llm.calls) == 3  # head + 2 sections (was: 1 call for everything)


def test_writer_section_path_parallel():
    f1 = _findings("SQ1", [("c1", "S1", "q1")])
    f2 = _findings("SQ2", [("c2", "S2", "q2")])
    llm = MockProvider(script=[HEAD, SEC_SAME, SEC_SAME])
    rep = run_writer([f1, f2], llm, "mock-agent", max_workers=3)
    assert rep.title == "Big Report"
    assert len(rep.sections) == 2
    assert rep.citations == ["S1", "S2"]


def test_writer_skips_empty_groups():
    f1 = _findings("SQ1", [("c1", "S1", "q1")])
    f2 = Findings(sub_question_id="SQ2", claims=[], sources=[])
    llm = MockProvider(script=[HEAD, SEC_SAME])
    rep = run_writer([f1, f2], llm, "mock-agent", max_workers=1)
    assert len(rep.sections) == 1  # no section for the empty group
    assert len(llm.calls) == 2


def test_writer_empty_findings_single_call_legacy():
    full = json.dumps({"title": "T", "summary": "S", "sections": [], "citations": []})
    llm = MockProvider(script=[full])
    rep = run_writer([Findings(sub_question_id="SQ1", claims=[], sources=[])], llm, "mock-agent")
    assert rep.title == "T" and rep.sections == []
    assert len(llm.calls) == 1
