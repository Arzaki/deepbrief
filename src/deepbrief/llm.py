"""Thin provider abstraction over OpenAI-compatible chat APIs + retry logic.

Providers: groq | openai | gemini | mock. All expose .complete(messages, model, **kw).
Retries with exponential backoff + jitter for 429 / transient errors, honours Retry-After.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

from .config import Settings


class LLMError(RuntimeError):
    def __init__(
        self,
        message: str,
        status: int | None = None,
        retry_after: float | None = None,
        retryable: bool | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
        # None = decide from status (429/5xx retryable). Set False to fail fast,
        # e.g. "request too large" 429s where retrying can never succeed.
        self.retryable = retryable


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


def compute_backoff(attempt: int, base: float = 1.0, cap: float = 30.0, retry_after: float | None = None) -> float:
    """Exponential backoff with jitter. Respects server Retry-After when larger."""
    exp = min(cap, base * (2**attempt))
    jitter = random.uniform(0, exp * 0.25)
    wait = exp + jitter
    if retry_after is not None:
        wait = max(wait, retry_after)
    return wait


def with_retries(fn, max_retries: int = 4, base: float = 1.0):
    """Call fn(); retry on LLMError with status 429 or >=500."""
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except LLMError as e:
            last = e
            if e.retryable is None:
                retryable = e.status is None or e.status == 429 or (e.status is not None and e.status >= 500)
            else:
                retryable = e.retryable
            if not retryable or attempt == max_retries:
                raise
            time.sleep(compute_backoff(attempt, base=base, retry_after=e.retry_after))
        except (TimeoutError, ConnectionError) as e:  # transient transport errors
            last = e
            if attempt == max_retries:
                raise LLMError(str(e)) from e
            time.sleep(compute_backoff(attempt, base=base))
    raise LLMError(str(last) if last else "retry loop exhausted")


class BaseProvider:
    name = "base"

    def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        raise NotImplementedError


class OpenAICompatibleProvider(BaseProvider):
    """Generic OpenAI-compatible chat provider (Groq, OpenAI, Gemini's OpenAI endpoint)."""

    def __init__(self, api_key: str, base_url: str, name: str = "openai-compatible"):
        from openai import OpenAI

        self.name = name
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, messages, model, temperature=0.0, seed=None, max_tokens=1000) -> LLMResponse:
        from openai import APIStatusError, APITimeoutError, RateLimitError

        def _call() -> LLMResponse:
            t0 = time.time()
            try:
                kwargs: dict[str, Any] = dict(
                    model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
                )
                if seed is not None:
                    kwargs["seed"] = seed
                resp = self._client.chat.completions.create(**kwargs)
            except RateLimitError as e:
                # "Request too large" 429s are about max_tokens, not rate: fail fast with a hint.
                too_large = "request too large" in str(e).lower()
                hint = " Hint: lower LLM_MAX_TOKENS (this plan caps output tokens per request)." if too_large else ""
                retry_after = None if too_large else _parse_retry_after(getattr(e, "response", None))
                raise LLMError(str(e) + hint, status=429, retry_after=retry_after, retryable=not too_large) from e
            except APIStatusError as e:
                retry_after = _parse_retry_after(getattr(e, "response", None))
                raise LLMError(str(e), status=getattr(e, "status_code", None), retry_after=retry_after) from e
            except APITimeoutError as e:
                raise LLMError(str(e), status=None) from e
            choice = resp.choices[0].message.content or ""
            usage = resp.usage
            return LLMResponse(
                text=choice,
                model=model,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                latency_s=time.time() - t0,
            )

        return with_retries(_call)


def _parse_retry_after(response: Any) -> float | None:
    try:
        if response is not None:
            v = response.headers.get("retry-after") or response.headers.get("Retry-After")
            if v:
                return float(v)
    except Exception:
        return None
    return None


class MockProvider(BaseProvider):
    """Deterministic stub for tests / offline CI. Returns canned JSON per call."""

    name = "mock"

    def __init__(self, script: list[str] | None = None):
        self.script = list(script or [])
        self.calls: list[dict] = []

    def complete(self, messages, model, temperature=0.0, seed=None, max_tokens=1000) -> LLMResponse:
        self.calls.append({"messages": messages, "model": model})
        if self.script:
            text = self.script.pop(0)
        else:
            text = _mock_default_reply(messages)
        return LLMResponse(text=text, model=model, input_tokens=10, output_tokens=10, latency_s=0.001)


def _mock_default_reply(messages: list[dict]) -> str:
    """Content-aware canned reply so offline runs exercise the full pipeline deterministically."""
    import json as _json

    blob = " ".join(str(m.get("content", "")) for m in messages).lower()
    if "sub-question" in blob and "findings" in blob and "tool" in blob:
        # Researcher turn: do one fake search then finish with empty findings on next call.
        if not any(m.get("role") == "assistant" for m in messages):
            return "TOOL: web_search | mock offline query"
        return _json.dumps({"sub_question_id": "SQ1", "claims": [], "sources": []})
    if "sub-questions" in blob or "decompose" in blob:
        return _json.dumps(
            {
                "sub_questions": [
                    {"id": "SQ1", "text": "mock aspect one", "rationale": "offline"},
                    {"id": "SQ2", "text": "mock aspect two", "rationale": "offline"},
                    {"id": "SQ3", "text": "mock aspect three", "rationale": "offline"},
                ]
            }
        )
    if "critique" in blob or "verdict" in blob and "findings" in blob:
        return '{"issues": [], "verdict": "pass"}'
    if "report" in blob or "citation" in blob:
        return _json.dumps(
            {
                "title": "Mock offline report",
                "summary": "Offline mock run with no live retrieval; content is placeholder.",
                "sections": [{"heading": "Overview", "body": "No verified claims in offline mode."}],
                "citations": [],
            }
        )
    return '{"issues": [], "verdict": "pass"}'


def make_provider(settings: Settings, script: list[str] | None = None) -> BaseProvider:
    name = settings.llm_provider.lower()
    if name == "mock":
        return MockProvider(script=script)
    if name == "groq":
        return OpenAICompatibleProvider(settings.groq_api_key, settings.groq_base_url, "groq")
    if name == "openai":
        return OpenAICompatibleProvider(settings.openai_api_key, "https://api.openai.com/v1", "openai")
    if name == "gemini":
        return OpenAICompatibleProvider(settings.gemini_api_key, settings.gemini_base_url, "gemini")
    raise ValueError(f"Unknown LLM_PROVIDER={settings.llm_provider!r} (expected groq|openai|gemini|mock)")


def make_judge_provider(settings: Settings) -> BaseProvider:
    """Judge ideally comes from a different family than the generator (configurable via env)."""
    if settings.judge_provider.lower() == "mock":
        return MockProvider()
    sub = Settings(
        llm_provider=settings.judge_provider,
        groq_api_key=settings.groq_api_key,
        openai_api_key=settings.openai_api_key,
        gemini_api_key=settings.gemini_api_key,
    )
    return make_provider(sub)
