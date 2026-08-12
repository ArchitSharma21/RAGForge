from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import httpx
import trafilatura
from ddgs import DDGS

from .config import get_settings
from .llm import GeminiGateway
from .security import is_safe_public_url


@dataclass(slots=True)
class WebPage:
    title: str
    url: str
    text: str
    snippet: str = ""


class WebSearchEngine:
    def __init__(self, gateway: GeminiGateway | None = None):
        self.gateway = gateway
        self.settings = get_settings()

    def search(self, query: str, provider: str = "Auto", max_results: int = 6) -> list[WebPage]:
        if provider == "Gemini Search":
            if not self.settings.enable_native_google_search:
                raise ValueError("Native Gemini Google Search is disabled by configuration")
            if not self.gateway:
                raise ValueError("Gemini Search requires a Gemini API key")
            answer, citations = self.gateway.native_web_search(query)
            return [WebPage(c["title"], c["url"], answer, answer[:500]) for c in citations]
        if provider == "Tavily" or (provider == "Auto" and self.settings.tavily_api_key):
            try:
                return self._tavily(query, max_results)
            except Exception:
                if provider == "Tavily":
                    raise
        return self._duckduckgo(query, max_results)

    def _duckduckgo(self, query: str, max_results: int) -> list[WebPage]:
        rows = list(DDGS().text(query, max_results=max_results))
        candidates = []
        for row in rows:
            url = row.get("href") or row.get("url") or ""
            if url and is_safe_public_url(url):
                candidates.append((row.get("title") or url, url, row.get("body") or ""))
        pages: list[WebPage] = []
        with ThreadPoolExecutor(max_workers=min(6, max(1, len(candidates)))) as pool:
            futures = {pool.submit(self._fetch, title, url, snippet): (title, url, snippet) for title, url, snippet in candidates}
            for future in as_completed(futures):
                try:
                    pages.append(future.result())
                except Exception:
                    title, url, snippet = futures[future]
                    pages.append(WebPage(title, url, snippet, snippet))
        return pages[:max_results]

    def _fetch(self, title: str, url: str, snippet: str) -> WebPage:
        headers = {"User-Agent": "RAGForge/1.0 (+https://huggingface.co/spaces)"}
        with httpx.Client(timeout=8.0, follow_redirects=False, headers=headers) as client:
            response = client.get(url)
            response.raise_for_status()
            ctype = response.headers.get("content-type", "")
            if "text" not in ctype and "html" not in ctype and "json" not in ctype:
                return WebPage(title, str(response.url), snippet, snippet)
            text = trafilatura.extract(response.text, include_links=False, include_tables=True) or snippet
            return WebPage(title, str(response.url), text[:18000], snippet)

    def _tavily(self, query: str, max_results: int) -> list[WebPage]:
        if not self.settings.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is not configured")
        payload = {
            "api_key": self.settings.tavily_api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_raw_content": True,
        }
        with httpx.Client(timeout=15.0) as client:
            resp = client.post("https://api.tavily.com/search", json=payload)
            resp.raise_for_status()
            data = resp.json()
        out = []
        for row in data.get("results", []):
            url = row.get("url", "")
            if not url or not is_safe_public_url(url):
                continue
            text = row.get("raw_content") or row.get("content") or ""
            out.append(WebPage(row.get("title") or url, url, text[:18000], row.get("content") or ""))
        return out
