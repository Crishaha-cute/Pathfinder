from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TOKEN_PATTERN = re.compile(r"[a-z0-9+#.]+", re.IGNORECASE)
MODEL_NAME = "all-MiniLM-L6-v2"
_sentence_model = None


@dataclass
class SearchIndex:
    embeddings: np.ndarray
    vocabulary: dict[str, int]
    idf: np.ndarray
    mode: str

    def search(self, query: str) -> np.ndarray:
        if self.mode == "sentence-transformers":
            model = _get_sentence_model()
            vector = model.encode([query], normalize_embeddings=True)[0]
            return np.clip(self.embeddings @ vector, 0, 1)
        query_vector = _tfidf_vector(query, self.vocabulary, self.idf)
        if not query_vector.any():
            return np.zeros(len(self.embeddings))
        return np.clip(self.embeddings @ query_vector, 0, 1)


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(str(text)) if len(token) > 1]


def _tfidf_vector(text: str, vocabulary: dict[str, int], idf: np.ndarray) -> np.ndarray:
    vector = np.zeros(len(vocabulary), dtype=np.float32)
    counts: dict[str, int] = {}
    for token in _tokens(text):
        if token in vocabulary:
            counts[token] = counts.get(token, 0) + 1
    for token, count in counts.items():
        vector[vocabulary[token]] = (1 + np.log(count)) * idf[vocabulary[token]]
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _build_tfidf(texts: list[str]) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    document_tokens = [set(_tokens(text)) for text in texts]
    vocabulary = {token: index for index, token in enumerate(sorted(set().union(*document_tokens)))}
    document_count = len(texts)
    frequencies = np.zeros(len(vocabulary), dtype=np.float32)
    for tokens in document_tokens:
        for token in tokens:
            frequencies[vocabulary[token]] += 1
    idf = np.log((1 + document_count) / (1 + frequencies)) + 1
    matrix = np.vstack([_tfidf_vector(text, vocabulary, idf) for text in texts])
    return matrix, vocabulary, idf


def _fingerprint(frame: pd.DataFrame) -> str:
    return hashlib.sha256("\n".join(frame["search_text"].astype(str)).encode("utf-8")).hexdigest()


def _get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer

        _sentence_model = SentenceTransformer(MODEL_NAME)
    return _sentence_model


def build_or_load_index(frame: pd.DataFrame, cache_path: Path, force_rebuild: bool = False) -> SearchIndex:
    metadata_path = cache_path.with_suffix(".json")
    fingerprint = _fingerprint(frame)
    if not force_rebuild and cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint:
            cached = np.load(cache_path, allow_pickle=False)
            vocabulary = json.loads(metadata["vocabulary"])
            return SearchIndex(cached["embeddings"], vocabulary, cached["idf"], metadata.get("mode", "tfidf"))

    try:
        model = _get_sentence_model()
        embeddings = model.encode(frame["search_text"].tolist(), normalize_embeddings=True)
        vocabulary = {}
        idf = np.array([], dtype=np.float32)
        mode = "sentence-transformers"
    except Exception:
        embeddings, vocabulary, idf = _build_tfidf(frame["search_text"].tolist())
        mode = "tfidf"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings, idf=idf)
    metadata_path.write_text(
        json.dumps({"fingerprint": fingerprint, "vocabulary": json.dumps(vocabulary), "mode": mode}),
        encoding="utf-8",
    )
    return SearchIndex(embeddings, vocabulary, idf, mode)
