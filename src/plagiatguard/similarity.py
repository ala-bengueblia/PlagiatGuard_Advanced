from difflib import SequenceMatcher

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import normalize


def lexical_similarity(text_a: str, text_b: str) -> float:
    a = normalize(text_a)
    b = normalize(text_b)

    if not a or not b:
        return 0.0

    try:
        word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=1,
        )
        word_matrix = word_vectorizer.fit_transform([a, b])
        word_score = float(
            cosine_similarity(
                word_matrix[0], word_matrix[1]
            )[0, 0]
        )
    except ValueError:
        word_score = 0.0

    try:
        char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=1,
        )
        char_matrix = char_vectorizer.fit_transform([a, b])
        char_score = float(
            cosine_similarity(
                char_matrix[0], char_matrix[1]
            )[0, 0]
        )
    except ValueError:
        char_score = 0.0

    sequence_score = SequenceMatcher(None, a, b).ratio()

    return max(
        0.0,
        min(
            1.0,
            0.55 * word_score
            + 0.30 * char_score
            + 0.15 * sequence_score,
        ),
    )


from functools import lru_cache

@lru_cache(maxsize=1)
def get_sentence_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def semantic_similarity(text_a: str, text_b: str):
    try:
        import numpy as np

        model = get_sentence_model()

        embeddings = model.encode(
            [text_a, text_b],
            normalize_embeddings=True
        )

        return float(np.dot(embeddings[0], embeddings[1]))

    except (ImportError, OSError, RuntimeError):
        return None

def combined_similarity(
    text_a: str,
    text_b: str,
    use_embeddings: bool = False,
) -> dict:

    lexical = lexical_similarity(text_a, text_b)

    semantic = (
        semantic_similarity(text_a, text_b)
        if use_embeddings else None
    )

    if semantic is None:
        score = lexical
    else:
        score = 0.65 * lexical + 0.35 * max(
            0.0, min(1.0, semantic)
        )

    return {
        "score": round(score, 4),
        "lexical_score": round(lexical, 4),
        "semantic_score": (
            None if semantic is None else round(semantic, 4)
        ),
    }