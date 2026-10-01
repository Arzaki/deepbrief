"""List model IDs actually available to your configured provider key.

Read-only inspection: one tiny GET to the provider's /models endpoint.
Your key stays local; only model IDs are printed (safe to paste into chat).
Run: uv run python scripts/check_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx

from deepbrief.config import Settings


def main() -> None:
    s = Settings()
    provider = s.llm_provider.lower()
    if provider == "mock":
        print("LLM_PROVIDER=mock: no live provider to query. Set LLM_PROVIDER=groq|openai|gemini first.")
        return
    bases = {
        "groq": (s.groq_base_url, s.groq_api_key),
        "openai": ("https://api.openai.com/v1", s.openai_api_key),
        "gemini": (s.gemini_base_url, s.gemini_api_key),
    }
    if provider not in bases:
        print(f"Unknown LLM_PROVIDER={s.llm_provider!r}")
        return
    base_url, key = bases[provider]
    if not key:
        print(f"No API key configured for provider={provider!r}. Set it in .env first.")
        return
    url = base_url.rstrip("/") + "/models"
    try:
        r = httpx.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=20)
        r.raise_for_status()
        ids = sorted(m.get("id", "") for m in r.json().get("data", []))
    except Exception as e:
        print(f"Could not list models for provider={provider!r}: {e}")
        return
    print(f"provider={provider} ({len(ids)} models):")
    for mid in ids:
        mark = ""
        if mid in (s.resolved_agent_model(), s.resolved_cheap_model()):
            mark = "  <-- currently configured default"
        print(f"  {mid}{mark}")


if __name__ == "__main__":
    main()
