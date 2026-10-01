"""Page fetching: httpx + trafilatura, cached by URL."""

from __future__ import annotations

import httpx
import trafilatura


def fetch_page(url: str, cache=None, trace=None, timeout: int = 12) -> dict:
    if cache is not None:
        hit = cache.get_json("page", url)
        if hit is not None:
            if trace:
                trace.log("tool", "fetch_page", url=url, cache_hit=True, chars=len(hit.get("text", "")))
            return hit
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "DeepBrief/0.1 research-bot"},
        ) as c:
            r = c.get(url)
            r.raise_for_status()
            html = r.text
        text = trafilatura.extract(html) or ""
        out = {"url": url, "title": "", "text": text[:12000]}
    except Exception as e:
        out = {"url": url, "title": "", "text": "", "error": str(e)}
    if cache is not None:
        cache.put_json("page", url, out)
    if trace:
        trace.log("tool", "fetch_page", url=url, cache_hit=False, chars=len(out.get("text", "")))
    return out
