
# src/plagiatguard/research_citations.py

import re


REFERENCE_HEADING = re.compile(
    r"(?im)^\s*(references|bibliography|bibliographie|"
    r"références|sources|works cited)\s*:?\s*$"
)

AUTHOR_YEAR = re.compile(
    r"\([A-ZÀ-ÖØ-Þ][\wÀ-ÿ'’\-]+"
    r"(?:\s+et al\.)?,?\s*(?:19|20)\d{2}[a-z]?\)"
)

NUMERIC_CITATION = re.compile(
    r"\[(\d{1,3}(?:\s*[,;\-]\s*\d{1,3})*)\]"
)

DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
URL = re.compile(r"https?://[^\s<>()]+", re.I)


def split_references(text):
    match = REFERENCE_HEADING.search(text or "")

    if not match:
        return {
            "reference_section_found": False,
            "body": text or "",
            "references": [],
        }

    body = text[:match.start()]
    reference_text = text[match.end():]

    # Découpage initial, à adapter aux styles bibliographiques réels.
    entries = re.split(
        r"\n\s*(?:\[\d+\]|\d+[.)])\s*",
        "\n" + reference_text,
    )

    references = [
        entry.strip()
        for entry in entries
        if len(entry.strip()) >= 15
    ]

    return {
        "reference_section_found": True,
        "body": body,
        "references": [
            {
                "text": entry,
                "dois": DOI.findall(entry),
                "urls": URL.findall(entry),
            }
            for entry in references
        ],
    }


def find_in_text_citations(body):
    citations = []

    for pattern, style in [
        (AUTHOR_YEAR, "author_year"),
        (NUMERIC_CITATION, "numeric"),
    ]:
        for match in pattern.finditer(body):
            citations.append({
                "text": match.group(),
                "start": match.start(),
                "end": match.end(),
                "style": style,
            })

    citations.sort(key=lambda item: item["start"])
    return citations


def find_quoted_segments(body):
    patterns = [
        r"“([^”]{20,})”",
        r'"([^"]{20,})"',
        r"«\s*([^»]{20,})\s*»",
    ]

    segments = []

    for pattern in patterns:
        for match in re.finditer(pattern, body, re.S):
            segments.append({
                "text": re.sub(r"\s+", " ", match.group(1)).strip(),
                "start": match.start(1),
                "end": match.end(1),
            })

    unique = {
        (item["start"], item["end"]): item
        for item in segments
    }

    return sorted(unique.values(), key=lambda item: item["start"])


def analyze_citations(text, matches=None):
    parsed = split_references(text)
    body = parsed["body"]

    citations = find_in_text_citations(body)
    quotes = find_quoted_segments(body)
    references = parsed["references"]

    # Ce contrôle indique un chevauchement explicite avec un passage cité.
    # Il ne vérifie pas à lui seul si la référence est exacte ou suffisante.
    annotated_matches = []

    for item in matches or []:
        source = item.get("source_passage", "")
        normalized_source = re.sub(
            r"\W+", " ", source.casefold()
        ).strip()

        overlaps_quote = any(
            re.sub(r"\W+", " ", quote["text"].casefold()).strip()
            in normalized_source
            for quote in quotes
        )

        annotated_matches.append({
            **item,
            "overlaps_explicit_quote": overlaps_quote,
            "citation_status": (
                "quote_overlap_review"
                if overlaps_quote
                else "citation_not_verified"
            ),
        })

    return {
        "reference_section_found": parsed["reference_section_found"],
        "reference_count": len(references),
        "in_text_citation_count": len(citations),
        "quoted_segment_count": len(quotes),
        "doi_count": sum(len(ref["dois"]) for ref in references),
        "reference_url_count": sum(len(ref["urls"]) for ref in references),
        "references": references,
        "in_text_citations": citations,
        "quoted_segments": quotes,
        "matched_passages": annotated_matches,
        "warning": (
            "L'analyse automatique des citations est indicative. "
            "Vérifiez la correspondance entre les citations dans le texte, "
            "les références bibliographiques et les sources originales."
        ),
    }
