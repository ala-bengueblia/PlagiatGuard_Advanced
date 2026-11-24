import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from plagiatguard.preprocessing import normalize, split_passages
from plagiatguard.similarity import lexical_similarity
from plagiatguard.citations import scan_citations


def test_normalize():
    assert normalize(" BONJOUR \n monde ") == "bonjour monde"


def test_identical_texts():
    text = "Le chat mange une pomme."
    assert lexical_similarity(text, text) > 0.95


def test_different_texts():
    score = lexical_similarity(
        "Le chat mange une pomme.",
        "Les avions traversent le ciel.",
    )
    assert score < 0.6


def test_split_passages():
    passages = split_passages(
        "texte " * 500,
        max_chars=100,
        overlap=10,
    )
    assert len(passages) > 1


def test_bibliography_heading():
    scan = scan_citations(
        "Article\nReferences\nSmith, 2020"
    )
    assert scan.bibliography_detected