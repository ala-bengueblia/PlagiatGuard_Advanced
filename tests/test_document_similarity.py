
from pathlib import Path
import sys
from itertools import combinations

# Ajouter src au chemin Python
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plagiatguard.extractors import extract_text
from plagiatguard.similarity import combined_similarity


CORPUS_DIR = ROOT / "data" / "corpus"

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".docx",
    ".odt"
}


def load_documents():
    documents = {}

    for file_path in CORPUS_DIR.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                content = file_path.read_bytes()
                text = extract_text(file_path.name, content)

                if text.strip():
                    documents[file_path.name] = text

            except Exception as exc:
                print(f"Erreur avec {file_path.name}: {exc}")

    return documents


def interpret_score(score):
    if score >= 0.80:
        return "Très forte similarité"
    elif score >= 0.60:
        return "Forte similarité"
    elif score >= 0.40:
        return "Similarité modérée"
    elif score >= 0.20:
        return "Faible similarité"
    return "Très faible similarité"


def main():
    documents = load_documents()

    print("\n========== DOCUMENTS CHARGÉS ==========")
    print(f"Nombre de documents : {len(documents)}")

    if len(documents) < 2:
        print("Il faut au moins deux documents.")
        return

    print("\n========== COMPARAISON DES DOCUMENTS ==========")

    for (name_a, text_a), (name_b, text_b) in combinations(
        documents.items(), 2
    ):
        result = combined_similarity(
            text_a,
            text_b,
            use_embeddings=True
        )

        score = result["score"]

        print("\n----------------------------------------")
        print(f"Document A : {name_a}")
        print(f"Document B : {name_b}")
        print(f"Similarité globale : {score * 100:.2f}%")
        print(f"Similarité lexicale : {result['lexical_score'] * 100:.2f}%")

        semantic = result["semantic_score"]

        if semantic is not None:
            print(f"Similarité sémantique : {semantic * 100:.2f}%")

        print(f"Interprétation : {interpret_score(score)}")

    print("\n========== TEST TERMINÉ ==========")


if __name__ == "__main__":
    main()
