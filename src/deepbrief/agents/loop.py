"""Plain-Python JSON tool-calling loop helpers shared by agents (no LangChain)."""

from __future__ import annotations

import json
import re


def extract_json(text: str) -> str:
    """Pull strict JSON out of an LLM reply (fenced code block or raw)."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def parse_json_strict(text: str) -> dict:
    return json.loads(extract_json(text))
