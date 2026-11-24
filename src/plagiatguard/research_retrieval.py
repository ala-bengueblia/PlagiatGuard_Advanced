
import json
import logging
import re
import unicodedata
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class PassageIndex:
    """
    Index de recherche de passages.
    Utilise FAISS si disponible, sinon TF-IDF.
    """

    def __init__(
        self,
        index_dir="artifacts/research_index",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "passages.faiss"
        self.metadata_path = self.index_dir / "passages.json"

        self.model_name = model_name
        self.passages = []
        self.metadata = []

        self.backend = None
        self.index = None
        self.model = None
        self.vectorizer = None
        self.tfidf_matrix = None

    @staticmethod
    def _clean_text(text):
        """
        Nettoie le texte sans modifier son contenu linguistique.
        Supprime notamment le BOM UTF-8.
        """
        text = str(text or "")

        text = text.replace("\ufeff", "")
        text = unicodedata.normalize("NFC", text)

        # Normalisation des espaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _prepare_directory(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def _save_metadata(self):
        data = {
            "passages": self.passages,
            "metadata": self.metadata,
            "backend": self.backend,
            "model_name": self.model_name
        }

        self._prepare_directory()

        with self.metadata_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    def _build_tfidf(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not self.passages:
            raise ValueError(
                "Aucun passage disponible pour construire l'index."
            )

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(self.passages)
        self.backend = "tfidf"

        logger.info("Index TF-IDF construit avec succès.")

    def build(self, passages, metadata=None):
        """
        Construit l'index à partir des passages et métadonnées.
        """

        if metadata is None:
            metadata = [{} for _ in passages]

        if len(metadata) != len(passages):
            raise ValueError(
                "Le nombre de métadonnées doit correspondre "
                "au nombre de passages."
            )

        cleaned_passages = []
        cleaned_metadata = []
        seen = set()

        for passage, passage_metadata in zip(passages, metadata):
            cleaned = self._clean_text(passage)

            if not cleaned:
                continue

            # Évite les doublons exacts tout en conservant
            # les métadonnées du premier passage rencontré.
            if cleaned in seen:
                continue

            seen.add(cleaned)
            cleaned_passages.append(cleaned)
            cleaned_metadata.append(passage_metadata)

        self.passages = cleaned_passages
        self.metadata = cleaned_metadata

        if not self.passages:
            raise ValueError(
                "Aucun passage valide à indexer."
            )

        self._prepare_directory()

        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Chargement du modèle : %s",
                self.model_name
            )

            self.model = SentenceTransformer(self.model_name)

            embeddings = self.model.encode(
                self.passages,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True
            )

            embeddings = np.asarray(
                embeddings,
                dtype=np.float32
            )

            dimension = embeddings.shape[1]

            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)

            serialized_index = faiss.serialize_index(self.index)

            self.index_path.write_bytes(
                serialized_index.tobytes()
            )

            self.backend = "faiss"

            logger.info(
                "Index FAISS construit et sauvegardé."
            )

        except Exception as error:
            logger.exception(
                "Construction FAISS impossible : %s",
                error
            )

            logger.warning("Activation du moteur TF-IDF.")

            self.index = None
            self.model = None

            self._build_tfidf()

            if self.index_path.exists():
                self.index_path.unlink()

        self._save_metadata()

        return {
            "passage_count": len(self.passages),
            "backend": self.backend,
            "index_path": str(self.index_path),
            "metadata_path": str(self.metadata_path)
        }

    def load(self):
        """
        Charge l'index sauvegardé.
        """

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {self.metadata_path}"
            )

        with self.metadata_path.open(
            "r",
            encoding="utf-8-sig"
        ) as file:
            data = json.load(file)

        self.passages = [
            self._clean_text(p)
            for p in data.get("passages", [])
            if self._clean_text(p)
        ]

        self.metadata = data.get("metadata", [])
        self.backend = data.get("backend")
        self.model_name = data.get(
            "model_name",
            self.model_name
        )

        if len(self.metadata) != len(self.passages):
            self.metadata = [
                {} for _ in self.passages
            ]

        if not self.passages:
            raise ValueError(
                "Le fichier d'index ne contient aucun passage."
            )

        if self.backend == "faiss":
            try:
                import faiss
                from sentence_transformers import SentenceTransformer

                if not self.index_path.exists():
                    raise FileNotFoundError(
                        f"Index FAISS introuvable : {self.index_path}"
                    )

                index_bytes = self.index_path.read_bytes()

                index_buffer = np.frombuffer(
                    index_bytes,
                    dtype=np.uint8
                )

                self.index = faiss.deserialize_index(
                    index_buffer
                )

                self.model = SentenceTransformer(
                    self.model_name
                )

                logger.info(
                    "Index FAISS chargé avec succès."
                )

            except Exception as error:
                logger.exception(
                    "Impossible de charger FAISS : %s",
                    error
                )

                logger.warning(
                    "Passage au moteur TF-IDF."
                )

                self.index = None
                self.model = None

                self._build_tfidf()

        else:
            self.index = None
            self.model = None
            self._build_tfidf()

        return {
            "passage_count": len(self.passages),
            "backend": self.backend
        }

    def search(self, query, top_k=5):
        """
        Recherche les passages les plus similaires à la requête.
        """

        query = self._clean_text(query)

        if not query:
            return []

        if not self.passages:
            raise RuntimeError(
                "L'index n'est pas chargé."
            )

        top_k = max(
            1,
            min(int(top_k), len(self.passages))
        )

        results = []

        if self.backend == "faiss":

            if self.model is None or self.index is None:
                self.load()

            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            query_embedding = np.asarray(
                query_embedding,
                dtype=np.float32
            )

            scores, indices = self.index.search(
                query_embedding,
                top_k
            )

            for score, index_id in zip(
                scores[0],
                indices[0]
            ):
                if index_id < 0:
                    continue

                results.append({
                    "passage": self.passages[int(index_id)],
                    "metadata": self.metadata[int(index_id)],
                    "score": float(score),
                    "backend": "faiss"
                })

        else:

            if (
                self.vectorizer is None
                or self.tfidf_matrix is None
            ):
                self._build_tfidf()

            query_vector = self.vectorizer.transform(
                [query]
            )

            similarities = (
                self.tfidf_matrix @ query_vector.T
            ).toarray().ravel()

            best_indices = np.argsort(
                similarities
            )[::-1][:top_k]

            for index_id in best_indices:
                results.append({
                    "passage": self.passages[int(index_id)],
                    "metadata": self.metadata[int(index_id)],
                    "score": float(similarities[index_id]),
                    "backend": "tfidf"
                })

        return results
