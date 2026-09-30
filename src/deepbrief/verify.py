"""Deterministic programmatic citation check (full impl in M5; signature stable since M1)."""

from __future__ import annotations

import difflib
import re


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def quote_found(quote: str, page_text: str, threshold: float = 0.85) -> bool:
    """True if normalized quote appears verbatim or fuzzy-matches above threshold."""
    if not quote or not page_text:
        return False
    nq, npg = normalize(quote), normalize(page_text)
    if nq in npg:
        return True
    # Fuzzy fallback: sliding window over page words.
    qw = nq.split()
    pgw = npg.split()
    if len(qw) > len(pgw):
        return difflib.SequenceMatcher(None, nq, npg).ratio() >= threshold
    best = 0.0
    for i in range(0, max(1, len(pgw) - len(qw) + 1)):
        window = " ".join(pgw[i : i + len(qw)])
        r = difflib.SequenceMatcher(None, nq, window).ratio()
        if r > best:
            best = r
            if best >= threshold:
                return True
    return best >= threshold


def verify_findings(findings, page_texts: dict[str, str]) -> list[dict]:
    """Return list of {claim, source_id, ok} flags. Deterministic, no LLM."""
    flags = []
    for f in findings:
        for c in f.claims if hasattr(f, "claims") else []:
            page = page_texts.get(c.source_id, "")
            flags.append(
                {
                    "claim": c.text[:120],
                    "source_id": c.source_id,
                    "ok": quote_found(c.supporting_quote, page),
                }
            )
    return flags
