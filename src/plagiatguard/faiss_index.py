import json

from .config import (
    CORPUS_DIR,
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
)
from .extractors import extract_text
from .preprocessing import split_passages


def load_dependencies():
    try:
        import faiss
        from sentence_transformers import SentenceTransformer

        return faiss, SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Installe faiss-cpu et sentence-transformers."
        ) from exc


def build_index():
    faiss, SentenceTransformer = load_dependencies()

    passages = []
    metadata = []

    for path in CORPUS_DIR.rglob("*"):
        if not path.is_file():
            continue

        try:
            text = extract_text(path.name, path.read_bytes())

            for i, passage in enumerate(
                split_passages(text)
            ):
                passages.append(passage)
                metadata.append({
                    "file": str(path.relative_to(CORPUS_DIR)),
                    "passage": i,
                    "text": passage,
                })

        except Exception:
            continue

    if not passages:
        raise RuntimeError(
            "Aucun passage trouvé dans le corpus."
        )

    model = SentenceTransformer(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    vectors = model.encode(
        passages,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    FAISS_INDEX_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    faiss.write_index(index, str(FAISS_INDEX_PATH))

    FAISS_META_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return len(passages)


def search_index(query: str, k: int = 5):
    faiss, SentenceTransformer = load_dependencies()

    if (
        not FAISS_INDEX_PATH.exists()
        or not FAISS_META_PATH.exists()
    ):
        raise RuntimeError(
            "Index absent. Construis-le d'abord."
        )

    index = faiss.read_index(str(FAISS_INDEX_PATH))
    metadata = json.loads(
        FAISS_META_PATH.read_text(encoding="utf-8")
    )

    model = SentenceTransformer(
        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, ids = index.search(
        vector,
        min(k, len(metadata)),
    )

    results = []

    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            results.append({
                "similarity": float(score),
                **metadata[int(idx)],
            })

    return results