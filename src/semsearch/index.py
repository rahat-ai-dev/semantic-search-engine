"""The search index: stores one vector per document and ranks them by cosine similarity."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse

from .embedders import Embedder, is_sparse, make_embedder

INDEX_ROOT = Path(__file__).resolve().parents[2] / "index"


@dataclass
class Hit:
    rank: int
    score: float  # cosine similarity, -1 .. 1 (higher = more similar)
    doc: dict


def cosine_scores(matrix, query_vec) -> np.ndarray:
    """Cosine similarity of one query against every document.

    Both sides are already L2-normalised (length 1), so
        cosine(a, b) = (a . b) / (|a| |b|) = a . b
    and scoring the whole corpus is a single matrix-vector product.
    """
    scores = matrix @ query_vec.T
    if sparse.issparse(scores):
        scores = scores.toarray()
    return np.asarray(scores).ravel()


class SearchIndex:
    def __init__(self, embedder: Embedder, docs: list[dict], matrix):
        self.embedder = embedder
        self.docs = docs
        self.matrix = matrix

    # -- build ---------------------------------------------------------------
    @classmethod
    def build(cls, docs: list[dict], embedder_name: str = "glove") -> "SearchIndex":
        embedder = make_embedder(embedder_name)
        texts = [f"{d['title']}. {d['text']}" for d in docs]
        embedder.fit(texts)
        matrix = embedder.encode(texts)
        return cls(embedder, docs, matrix)

    # -- search --------------------------------------------------------------
    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = self.embedder.encode([query])
        scores = cosine_scores(self.matrix, q)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]  # k best, unordered (fast)
        top = top[np.argsort(-scores[top])]  # sort only those k
        return [Hit(r + 1, float(scores[i]), self.docs[i]) for r, i in enumerate(top)]

    # -- save / load -----------------------------------------------------------
    def save(self, name: str | None = None) -> Path:
        folder = INDEX_ROOT / (name or self.embedder.name)
        folder.mkdir(parents=True, exist_ok=True)
        if is_sparse(self.matrix):
            sparse.save_npz(folder / "matrix.npz", self.matrix)
        else:
            np.save(folder / "matrix.npy", self.matrix)
        joblib.dump(self.embedder, folder / "embedder.joblib")
        with open(folder / "docs.jsonl", "w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        return folder

    @classmethod
    def load(cls, name: str) -> "SearchIndex":
        folder = INDEX_ROOT / name
        if not folder.exists():
            raise FileNotFoundError(f"No index '{name}'. Run `python -m semsearch.cli build --embedder {name}`.")
        embedder = joblib.load(folder / "embedder.joblib")
        if (folder / "matrix.npz").exists():
            matrix = sparse.load_npz(folder / "matrix.npz")
        else:
            matrix = np.load(folder / "matrix.npy")
        with open(folder / "docs.jsonl", encoding="utf-8") as f:
            docs = [json.loads(line) for line in f]
        return cls(embedder, docs, matrix)
