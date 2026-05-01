"""
Web Search Tool - allows Researcher Agent to gather information from the web

Support 2 modes:
1. Scrape a specific URL (BeautifulSoup)
2. Search via Bing Search API (if you have the key)

This tool is registered with FoundryChatClient agent as a @tool decorator
"""
from __future__ import annotations

import json
import logging
from typing import Callable
# from typing import Optional
# from urllib.parse import urlparse

from agent_framework import tool

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
TIMEOUT = 15  # seconds
MAX_CONTENT_CHARS = 6000  # Avoid exceeding context window

@tool(approval_mode="never_require")
def scrape_url(url: str) -> str:
    """
    Scrape text content from a URL

    Args:
        url: URL for data collection

    Returns:
        JSON String: {"url": ..., "title": ..., "content": ..., "error": ...}
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script, style, nav, footer
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            tag.decompose()

        title = soup.title.get_text(strip=True) if soup.title else ""

        # Get the main content
        main = soup.find("main") or soup.find("article") or soup.find("div", class_="content")
        body = main if main else soup.body
        content = body.get_text(separator="\n", strip=True) if body else ""

        # Trim if it's to long
        if len(content) > MAX_CONTENT_CHARS:
            content = content[:MAX_CONTENT_CHARS] + "\n...[truncated]"

        return json.dumps(
            {
                "url": url,
                "title": title,
                "content": content,
                "error": None
            },
            ensure_ascii=False,
        )

    except requests.exceptions.RequestException as e:
        logger.warning(f"Scrape failed for {url}: {e}")
        return json.dumps({"url": url, "title": "", "content": "", "error": str(e)}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Unexpected error scraping {url}: {e}")
        return json.dumps({"url": url, "title": "", "content": "", "error": str(e)}, ensure_ascii=False)

# Todo: Integrate Azure AI Search
@tool(approval_mode="never_require")
def search_admission_info(query: str, num_results: int = 5) -> str:
    """
    Search for admissions information on the web (demo: return mock data if Bing key is not availability)

    In production: Integrate Bing Search API or Azure AI Search

    Args:
        query: Search question (Ex: "điểm chuẩn Bách Khoa Hà Nội 2025")
        num_results: Maximum number of results

    Returns:
        The JSON string containing the search results
    """
    import os

    bing_key = os.environ.get("BING_SEARCH_API_KEY")

    if bing_key:
        return _bing_search(query, bing_key, num_results)
    else:
        # Fallback: Return instruction to direct the scrape agent
        logger.info("No Bing key found, returning search suggestion.")
        suggested_urls = _suggest_urls_for_query(query)
        return json.dumps(
            {
                "query": query,
                "note": "Bing API key not configured. Suggested URLs to scrape:",
                "suggested_urls": suggested_urls,
            },
            ensure_ascii=False,
        )


def _bing_search(query: str, api_key: str, num_results: int) -> str:
    """Calling Bing Search API."""
    endpoint = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query, "count": num_results, "mkt": "vi-VN", "safeSearch": "Moderate"}

    try:
        resp = requests.get(endpoint, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = [
            {"title": item.get("name", ""), "url": item.get("url", ""), "snippet": item.get("snippet", "")}
            for item in data.get("webPages", {}).get("value", [])
        ]
        return json.dumps({"query": query, "results": results}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"query": query, "results": [], "error": str(e)}, ensure_ascii=False)


def _suggest_urls_for_query(query: str) -> list[str]:
    """
    Suggest URL to scrape based on query keywords.
    Used when there is no Bing API key
    """
    query_lower = query.lower()
    urls = []

    if "điểm chuẩn" in query_lower or "benchmark" in query_lower:
        urls.append("https://vnexpress.net/diem-chuan-hon-200-dai-hoc-nam-2025-cap-nhat-nhanh-chi-tiet-chinh-xac-nhat-4929038.html")
        urls.append("https://baochinhphu.vn/cap-nhat-diem-chuan-cua-cac-truong-dai-hoc-2025-102250819160544674.htm")

    if "khoa học tự nhiên" in query_lower or "hcmus" in query_lower:
        urls.append("https://tuyensinh.hcmus.edu.vn/2026-phuong-huong-tuyen-sinh-trinh-do-dai-hoc-du-kien/")
    
    if "bách khoa hcm" in query_lower or "hcmus" in query_lower:
        urls.append("https://hcmut.edu.vn/tintuc/cong-bo-thong-tin-tuyen-sinh-dai-hoc-chinh-quy-nam-2026")

    if not urls:
        urls = [
            "https://vnexpress.net/giao-duc/tuyen-sinh",
            "https://dantri.com.vn/giao-duc/tuyen-sinh.htm",
        ]

    return urls

# List of tools to import into researcher.py
ALL_SEARCH_TOOLS = [ scrape_url, search_admission_info ]