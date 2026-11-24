import re
from plagiatguard.research_citations import analyze_citations
from plagiatguard.research_web import (
    search_web,
    compare_passages,
)


def segment_text(
    text,
    max_words=180,
    overlap=25,
    min_words=10,
):
    words = re.findall(r"\S+", text or "")

    if not words:
        return []

    if max_words <= 0:
        raise ValueError(
            "max_words doit être supérieur à zéro."
        )

    if min_words <= 0:
        raise ValueError(
            "min_words doit être supérieur à zéro."
        )

    if overlap < 0 or overlap >= max_words:
        raise ValueError(
            "overlap doit être compris entre 0 et max_words - 1."
        )

    passages = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        passage = " ".join(words[start:end]).strip()

        if len(words[start:end]) >= min_words:
            passages.append(passage)

        if end == len(words):
            break

        start = end - overlap

    return passages


class ResearchPipeline:

    def __init__(
        self,
        passage_index,
        top_k=5,
        local_threshold=0.25,
    ):
        self.passage_index = passage_index
        self.top_k = top_k
        self.local_threshold = local_threshold

    def analyze(self, document_text, enable_web=False):

        passages = segment_text(document_text)

        local_matches = []

        # 1. Recherche locale FAISS / TF-IDF.
        if self.passage_index:

            for position, passage in enumerate(passages):

                hits = self.passage_index.search(
                    passage,
                    top_k=self.top_k,
                )

                for hit in hits:

                    score = float(
                        hit.get("score", 0.0)
                    )

                    if score < self.local_threshold:
                        continue

                    metadata = hit.get("metadata", {})

                    if isinstance(metadata, dict):
                        sources = metadata.get(
                            "sources",
                            [metadata],
                        )

                    elif isinstance(metadata, list):
                        sources = metadata

                    else:
                        sources = []

                    local_matches.append({
                        "submitted_passage": passage,
                        "source_passage": hit["passage"],
                        "score": score,
                        "source": sources,
                        "engine": hit.get(
                            "backend",
                            getattr(
                                self.passage_index,
                                "backend",
                                "unknown",
                            ),
                        ),
                        "segment_index": position,
                        "requires_manual_review": True,
                    })

        # 2. Regroupement des correspondances locales.
        grouped_matches = {}

        for hit in local_matches:

            key = (
                hit["submitted_passage"],
                hit["source_passage"],
            )

            if key not in grouped_matches:
                grouped_matches[key] = hit.copy()

            else:
                existing = grouped_matches[key]

                existing["score"] = max(
                    existing["score"],
                    hit["score"],
                )

                for source in hit["source"]:
                    if source not in existing["source"]:
                        existing["source"].append(source)

        local_matches = list(
            grouped_matches.values()
        )

        # 3. Recherche Web indépendante.
        web_sources = []
        web_matches = []
        web_error = None

        if enable_web:

            for passage in passages[:3]:

                try:
                    results = search_web(
                        passage[:900],
                        max_results=5,
                    )

                    web_sources.extend(results)

                except Exception as exc:
                    web_error = str(exc)

            # Déduplication par URL.
            unique_sources = {}

            for source in web_sources:
                url = source.get("url")

                if url:
                    unique_sources[url] = source

            web_sources = list(
                unique_sources.values()
            )

            web_matches = compare_passages(
                passages,
                web_sources,
            )

        # 4. Analyse des citations.
        citation_inputs = []

        for hit in local_matches:
            citation_inputs.append({
                **hit,
                "match_origin": "local_corpus",
            })

        for hit in web_matches:
            citation_inputs.append({
                **hit,
                "match_origin": "web",
            })

        citation_analysis = analyze_citations(
            document_text,
            citation_inputs,
        )

        # 5. Rapport final.
        return {
            "status": "completed",
            "segment_count": len(passages),

            "local_match_count": len(local_matches),

            "web_enabled": enable_web,
            "web_source_count": len(web_sources),
            "web_match_count": len(web_matches),
            "web_error": web_error,

            "local_matches": sorted(
                local_matches,
                key=lambda item: item["score"],
                reverse=True,
            ),

            "web_sources": web_sources,

            "web_matches": web_matches,

            "citation_analysis": citation_analysis,

            "interpretation": (
                "Les scores indiquent une similarité entre passages. "
                "Ils ne constituent pas une décision automatique de plagiat."
            ),
        }