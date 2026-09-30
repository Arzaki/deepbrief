"""Eval metrics: coverage, citation accuracy, faithfulness + cost/latency passthrough."""

from __future__ import annotations


def coverage_score(judgments) -> float:
    if not judgments:
        return 0.0
    return sum(1 for j in judgments if j.present) / len(judgments)


def entailment_scores(judgments) -> dict:
    if not judgments:
        return {"citation_accuracy": 0.0, "faithfulness": 0.0}
    sup = sum(1 for j in judgments if j.verdict == "supported")
    part = sum(1 for j in judgments if j.verdict == "partially")
    n = len(judgments)
    return {
        "citation_accuracy": sup / n,
        "faithfulness": (sup + 0.5 * part) / n,
        "supported": sup,
        "partial": part,
        "unsupported": n - sup - part,
    }
