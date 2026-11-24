import re
from dataclasses import dataclass


@dataclass
class CitationScan:
    citation_spans: list[tuple[int, int]]
    bibliography_spans: list[tuple[int, int]]
    bibliography_detected: bool


BIB_HEADINGS = re.compile(
    r"(?im)^\s*(references|bibliography|works cited|"
    r"bibliographie|références|references bibliographiques)\s*$"
)

IN_TEXT_CITATIONS = [
    re.compile(
        r"\([A-ZÀ-ÖØ-Þ][\wÀ-ÿ'’-]+"
        r"(?:\s+et al\.)?,?\s*(?:19|20)\d{2}[a-z]?\)"
    ),
    re.compile(r"\[[0-9]{1,3}(?:\s*[,;–-]\s*[0-9]{1,3})*\]"),
    re.compile(
        r"\([A-ZÀ-ÖØ-Þ][\wÀ-ÿ'’-]+,\s*(?:19|20)\d{2}\)"
    ),
]


def scan_citations(text: str) -> CitationScan:
    citation_spans = []

    for pattern in IN_TEXT_CITATIONS:
        citation_spans.extend(
            (match.start(), match.end())
            for match in pattern.finditer(text or "")
        )

    heading = BIB_HEADINGS.search(text or "")
    bibliography_spans = []

    if heading:
        bibliography_spans.append((heading.start(), len(text)))

    return CitationScan(
        citation_spans=citation_spans,
        bibliography_spans=bibliography_spans,
        bibliography_detected=bool(heading),
    )


def citation_overlap_ratio(text: str) -> float:
    scan = scan_citations(text)

    if not text:
        return 0.0

    covered = sum(
        end - start for start, end in scan.citation_spans
    )
    covered += sum(
        end - start for start, end in scan.bibliography_spans
    )

    return min(covered / len(text), 1.0)