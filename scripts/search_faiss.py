
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from plagiatguard.research_retrieval import PassageIndex


INDEX_DIR = ROOT / "artifacts" / "research_index"


def main():
    passage_index = PassageIndex(
        index_dir=str(INDEX_DIR)
    )

    try:
        info = passage_index.load()

        print("\n========== INDEX CHARGÉ ==========")
        print(f"Nombre de passages : {info['passage_count']}")
        print(f"Moteur : {info['backend']}")

    except Exception as error:
        print(f"Erreur de chargement : {error}")
        return

    print("\nSaisir 'exit' pour quitter.")

    while True:
        query = input("\nVotre requête : ").strip()

        if query.lower() in {"exit", "quit", "q"}:
            break

        if not query:
            print("Veuillez saisir une requête.")
            continue

        try:
            results = passage_index.search(
                query=query,
                top_k=5
            )

            if not results:
                print("Aucun résultat trouvé.")
                continue

            print("\n========== RÉSULTATS ==========")

            for rank, result in enumerate(results, start=1):
                metadata = result.get("metadata", {})

                print(f"\nRésultat {rank}")
                print(f"Score : {result['score']:.4f}")
                print(f"Moteur : {result['backend']}")
                print(f"Source : {metadata.get('source', 'Inconnue')}")
                print(f"Passage : {result['passage'][:700]}")

        except Exception as error:
            print(f"Erreur pendant la recherche : {error}")


if __name__ == "__main__":
    main()
