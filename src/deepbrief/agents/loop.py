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


def parse_json_lenient(text: str) -> dict:
    """Strict first; fall back to strict=False (allows literal newlines/tabs in strings).

    Fixes the control-character class of model JSON errors. Cannot fix unescaped
    double quotes inside strings — those need a retry with the error fed back.
    """
    raw = extract_json(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(raw, strict=False)


def trim_transcript(transcript: list[dict], keep_tail: int = 6, max_chars: int = 12000) -> list[dict]:
    """Bound request size for long tool loops.

    Always keeps the system prompt + original question, then the newest tail;
    drops oldest middle exchanges first. Prevents 413 request-too-large deaths
    (observed 2026-10-01: baseline accumulating 4KB fetches died at 444s with 413)
    and keeps late calls fast (smaller inputs queue cheaper too). FINAL payloads
    are self-contained, so dropping middle tool output is safe.
    """
    if len(transcript) <= 2 + keep_tail:
        head, tail = transcript[:2], transcript[2:]
    else:
        head, tail = transcript[:2], transcript[-keep_tail:]
    total = sum(len(str(m.get("content", ""))) for m in head + tail)
    tail = list(tail)
    while total > max_chars and len(tail) > 2:
        dropped = tail.pop(0)
        total -= len(str(dropped.get("content", "")))
    return head + tail


def retry_logger(trace, agent_name: str):
    """Build an on_retry callback logging otherwise-invisible backoff stalls."""
    if trace is None:
        return None

    def _log(exc, wait_s: float, attempt: int) -> None:
        trace.log(
            "llm_retry",
            agent_name,
            status=getattr(exc, "status", None),
            error=str(exc)[:200],
            wait_s=round(wait_s, 2),
            attempt=attempt,
        )

    return _log


def complete_json_with_retry(
    llm,
    messages: list[dict],
    model: str,
    pydantic_model,
    trace,
    agent_name: str,
    temperature: float = 0.0,
    seed: int | None = None,
    max_tokens: int = 1000,
    retries: int = 2,
    on_retry=None,
):
    """One LLM call returning validated JSON, with error-feedback retries.

    Shared by writer/critic (planner has its own equivalent loop). Raises a clear
    RuntimeError instead of leaking a raw JSONDecodeError after retries run out.
    """
    last_err = ""
    thread = list(messages)
    for attempt in range(retries + 1):
        if attempt > 0:
            thread = thread + [
                {
                    "role": "user",
                    "content": (
                        f"Your previous reply was not valid JSON ({last_err}). "
                        "Reply again with ONLY the JSON object: escape any double quotes "
                        "inside strings and keep it to one object."
                    ),
                }
            ]
        resp = llm.complete(thread, model, temperature=temperature, seed=seed, max_tokens=max_tokens, on_retry=on_retry)
        if trace is not None:
            trace.log_llm(agent_name, model, resp.input_tokens, resp.output_tokens, resp.latency_s)
        try:
            return pydantic_model.model_validate(parse_json_lenient(resp.text))
        except Exception as e:  # JSON error or schema validation error: retry with feedback
            last_err = str(e)[:300]
    raise RuntimeError(f"{agent_name} failed to produce valid JSON after {retries + 1} attempts: {last_err}")
