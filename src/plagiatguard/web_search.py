import requests
from bs4 import BeautifulSoup

from .config import ENABLE_WEB_SEARCH, TAVILY_API_KEY


def search_web(query: str, max_results: int = 5) -> list[dict]:
    if not ENABLE_WEB_SEARCH:
        return []

    if not TAVILY_API_KEY:
        raise RuntimeError(
            "La clé TAVILY_API_KEY est absente."
        )

    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": False,
        },
        timeout=20,
    )

    response.raise_for_status()

    results = response.json().get("results", [])

    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("content", ""),
        }
        for item in results
    ]


def fetch_page(url: str) -> dict:
    if not url.startswith(("https://", "http://")):
        raise ValueError("URL invalide.")

    response = requests.get(
        url,
        timeout=10,
        headers={"User-Agent": "PlagiatGuard/1.0"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    return {
        "url": response.url,
        "title": (
            soup.title.get_text(" ", strip=True)
            if soup.title else ""
        ),
        "text": soup.get_text(" ", strip=True)[:100_000],
    }