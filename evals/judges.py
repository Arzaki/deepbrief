"""Eval judges: coverage + entailment. Strict JSON validated by Pydantic. Offline-capable."""

from __future__ import annotations

from deepbrief.agents.loop import parse_json_strict
from deepbrief.llm import BaseProvider
from deepbrief.schemas import CoverageJudgment, CoverageResult, EntailmentJudgment, EntailmentResult


def judge_coverage(
    report_md: str, facts: list[str], llm: BaseProvider, model: str, max_tokens: int = 1000
) -> CoverageResult:
    if llm.name == "mock":
        # Offline heuristic: fact present if >=60% of its significant words appear.
        judgments = []
        low = report_md.lower()
        for fact in facts:
            words = [w.strip(".,;:()\"'").lower() for w in fact.split() if len(w) > 3]
            hit = sum(1 for w in words if w in low)
            present = bool(words) and hit / len(words) >= 0.6
            judgments.append(
                CoverageJudgment(fact=fact, present=present, evidence="offline-heuristic" if present else "")
            )
        return CoverageResult(judgments=judgments)
    judgments = []
    for fact in facts:
        prompt = (
            f"Report:\n{report_md[:6000]}\n\nFact: {fact}\n"
            'Return STRICT JSON: {"fact": "...", "present": true|false, "evidence": "..."}'
        )
        resp = llm.complete([{"role": "user", "content": prompt}], model, temperature=0.0, max_tokens=max_tokens)
        judgments.append(CoverageJudgment.model_validate(parse_json_strict(resp.text)))
    return CoverageResult(judgments=judgments)


def judge_entailment(claims: list[dict], llm: BaseProvider, model: str, max_tokens: int = 1000) -> EntailmentResult:
    if llm.name == "mock":
        from deepbrief.verify import quote_found

        judgments = []
        for i, c in enumerate(claims):
            ok = quote_found(c.get("supporting_quote", ""), c.get("page_text", ""))
            judgments.append(
                EntailmentJudgment(
                    claim_index=i,
                    verdict="supported" if ok else "unsupported",
                    reason="offline: programmatic quote check",
                )
            )
        return EntailmentResult(judgments=judgments)
    judgments = []
    for i, c in enumerate(claims):
        prompt = (
            f"Claim: {c.get('text')}\nQuote: {c.get('supporting_quote')}\n"
            f"Source excerpt: {c.get('page_text', '')[:3000]}\n"
            'Return STRICT JSON: {"claim_index": N, "verdict": "supported|partially|unsupported", "reason": "..."}'
        )
        resp = llm.complete([{"role": "user", "content": prompt}], model, temperature=0.0, max_tokens=max_tokens)
        d = parse_json_strict(resp.text)
        d["claim_index"] = i
        judgments.append(EntailmentJudgment.model_validate(d))
    return EntailmentResult(judgments=judgments)
