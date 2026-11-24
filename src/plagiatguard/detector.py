from pathlib import Path

from .config import CORPUS_DIR, USE_EMBEDDINGS
from .extractors import extract_text
from .similarity import combined_similarity
from .citations import citation_overlap_ratio
from .preprocessing import split_passages


SUPPORTED = {
    ".txt", ".md", ".pdf", ".docx", ".odt",
    ".html", ".htm", ".xml", ".json", ".csv",
    ".tsv", ".py", ".js", ".ts", ".java",
    ".cpp", ".c", ".h", ".ipynb",
}


def analyze_text(
    text: str,
    filename: str = "document.txt",
    threshold: float = 0.55,
) -> dict:

    passages = split_passages(text)
    matches = []

    for path in CORPUS_DIR.rglob("*"):
        if (
            not path.is_file()
            or path.suffix.lower() not in SUPPORTED
        ):
            continue

        try:
            other_text = extract_text(
                path.name, path.read_bytes()
            )
        except Exception:
            continue

        best = {"score": 0.0}

        for passage in passages[:200]:
            result = combined_similarity(
                passage,
                other_text,
                USE_EMBEDDINGS,
            )

            if result["score"] > best["score"]:
                best = result

        if best["score"] >= threshold:
            matches.append({
                "source": str(path.relative_to(CORPUS_DIR)),
                **best,
            })

    matches.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    citation_ratio = citation_overlap_ratio(text)
    max_score = matches[0]["score"] if matches else 0.0

    adjusted_score = max(
        0.0,
        max_score - min(0.20, citation_ratio * 0.20),
    )

    return {
        "filename": filename,
        "character_count": len(text),
        "passage_count": len(passages),
        "max_similarity": round(max_score, 4),
        "citation_overlap_ratio": round(citation_ratio, 4),
        "adjusted_similarity": round(adjusted_score, 4),
        "threshold": threshold,
        "flagged": adjusted_score >= threshold,
        "matches": matches[:20],
        "notice": (
            "Indicateur de similarité uniquement. "
            "Une vérification humaine est nécessaire."
        ),
    }


def analyze_bytes(filename, content, threshold=0.55):
    if Path(filename).suffix.lower() not in SUPPORTED:
        raise ValueError("Extension non autorisée.")

    text = extract_text(filename, content)

    return text, analyze_text(
        text,
        filename,
        threshold,
    )