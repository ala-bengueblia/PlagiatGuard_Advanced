
import sys
import logging
import unicodedata
from pathlib import Path

# Racine du projet
ROOT = Path(__file__).resolve().parents[1]

# Ajouter src au chemin Python
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from plagiatguard.research_retrieval import PassageIndex
from plagiatguard.research_pipeline import segment_text
from plagiatguard.extractors import extract_text


CORPUS_DIR = ROOT / "data" / "corpus"
INDEX_DIR = ROOT / "artifacts" / "research_index"

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".pdf",
    ".docx",
    ".odt"
}

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


def clean_text(text):
    """Nettoie le texte avant segmentation."""
    text = str(text or "")
    text = text.replace("\ufeff", "")
    text = unicodedata.normalize("NFC", text)
    return text.strip()


def main():
    passages = []
    metadata = []

    if not CORPUS_DIR.exists():
        raise FileNotFoundError(
            f"Le dossier corpus n'existe pas : {CORPUS_DIR}"
        )

    files = sorted([
        file for file in CORPUS_DIR.rglob("*")
        if file.is_file()
        and file.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not files:
        print("Aucun document trouvé dans data/corpus.")
        return

    print(f"Documents trouvés : {len(files)}")

    for file_path in files:
        try:
            logger.info("Traitement : %s", file_path.name)

            file_bytes = file_path.read_bytes()

            # Lecture du document
            if file_path.suffix.lower() == ".txt":
                text = file_bytes.decode(
                    "utf-8-sig",
                    errors="replace"
                )
            else:
                text = extract_text(
                    file_path.name,
                    file_bytes
                )

            text = clean_text(text)

            if not text:
                logger.warning(
                    "Document vide ou illisible : %s",
                    file_path.name
                )
                continue

            # Segmentation
            document_passages = segment_text(text)

            valid_count = 0

            for passage in document_passages:
                passage = clean_text(passage)

                if not passage:
                    continue

                passages.append(passage)

                metadata.append({
                    "source": file_path.name,
                    "path": str(file_path.relative_to(ROOT))
                })

                valid_count += 1

            logger.info(
                "%s : %s passage(s) valide(s)",
                file_path.name,
                valid_count
            )

        except Exception as error:
            logger.exception(
                "Erreur pendant le traitement de %s : %s",
                file_path.name,
                error
            )

    if not passages:
        print("Aucun passage valide n'a été extrait.")
        return

    print(f"\nNombre total de passages extraits : {len(passages)}")

    passage_index = PassageIndex(
        index_dir=str(INDEX_DIR)
    )

    result = passage_index.build(
        passages=passages,
        metadata=metadata
    )

    print("\n========== RÉSULTAT ==========")
    print(f"Passages indexés : {result['passage_count']}")
    print(f"Moteur utilisé : {result['backend']}")
    print(f"Dossier : {INDEX_DIR}")

    if result["backend"] == "faiss":
        print("\nINDEX FAISS CONSTRUIT AVEC SUCCÈS.")
    else:
        print("\nINDEX TF-IDF CONSTRUIT (FAISS indisponible).")


if __name__ == "__main__":
    main()
