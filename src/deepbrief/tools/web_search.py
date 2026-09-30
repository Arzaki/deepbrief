"""Web search: Tavily when TAVILY_API_KEY is set, else DuckDuckGo. Results cached."""

from __future__ import annotations

from typing import Any


def web_search(query: str, max_results: int = 5, cache=None, trace=None) -> list[dict[str, Any]]:
    if cache is not None:
        hit = cache.get_json("search", query)
        if hit is not None:
            if trace:
                trace.log("tool", "web_search", query=query, cache_hit=True, n=len(hit))
            return hit
    import os

    results: list[dict[str, Any]] = []
    try:
        if os.environ.get("TAVILY_API_KEY"):
            from tavily import TavilyClient

            client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
            resp = client.search(query, max_results=max_results, include_answer=False)
            for r in resp.get("results", []):
                results.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("content", "")[:500],
                    }
                )
        else:
            from ddgs import DDGS

            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(
                        {
                            "title": r.get("title", ""),
                            "url": r.get("href", ""),
                            "snippet": r.get("body", "")[:500],
                        }
                    )
    except Exception as e:  # offline / no key -> empty, caller handles gracefully
        results = [{"title": "", "url": "", "snippet": f"SEARCH_ERROR: {e}"}]
    if cache is not None:
        cache.put_json("search", query, results)
    if trace:
        trace.log("tool", "web_search", query=query, cache_hit=False, n=len(results))
    return results
