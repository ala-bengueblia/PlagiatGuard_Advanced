
# src/plagiatguard/research_web.py

import os
import re
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize(text):
    return re.sub(r"\s+", " ", text or "").strip()


def search_web(query, max_results=5):
    if os.getenv("ENABLE_WEB_SEARCH", "false").lower() not in {
        "true", "1", "yes"
    }:
        return []

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY n'est pas configurée.")

    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query[:900],
            "search_depth": "basic",
            "max_results": min(max(1, max_results), 10),
            "include_raw_content": False,
        },
        timeout=(5, 20),
    )
    response.raise_for_status()

    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("content", ""),
            "provider": "tavily",
        }
        for item in response.json().get("results", [])
    ]


def compare_passages(submitted_passages, sources):
    matches = []

    for submitted in submitted_passages:
        submitted = normalize(submitted)
        if len(submitted) < 40:
            continue

        for source in sources:
            snippet = normalize(source.get("snippet", ""))
            if len(snippet) < 20:
                continue

            vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            try:
                matrix = vectorizer.fit_transform([submitted, snippet])
                score = float(
                    cosine_similarity(matrix[0:1], matrix[1:2])[0, 0]
                )
            except ValueError:
                continue

            if score > 0:
                matches.append({
                    "submitted_passage": submitted,
                    "source_passage": snippet,
                    "score": score,
                    "source_title": source.get("title", ""),
                    "source_url": source.get("url", ""),
                    "provider": source.get("provider", "unknown"),
                    "requires_manual_review": True,
                })

    return sorted(matches, key=lambda item: item["score"], reverse=True)
