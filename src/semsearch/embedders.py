"""Embedding backends: turn text into vectors so that "similar meaning" = "similar direction".

Three interchangeable backends share one tiny interface (`fit`, `encode`):

* tfidf   - classical bag-of-words baseline (sparse). Matches *words*, not meaning.
* glove   - pretrained GloVe word vectors, averaged with IDF weights (dense, 100-d).
* minilm  - a real Transformer (all-MiniLM-L6-v2) via sentence-transformers (dense, 384-d).
            Needs internet access to huggingface.co the first time.

Every `encode()` returns L2-normalised rows, so cosine similarity is just a dot product.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from . import data


class Embedder:
    name = "base"

    def fit(self, texts: list[str]) -> "Embedder":
        return self

    def encode(self, texts: list[str]):  # -> np.ndarray | sparse matrix
        raise NotImplementedError


class TfidfEmbedder(Embedder):
    """Sparse TF-IDF vectors (the Day 92 baseline). Rare shared words get high weight."""

    name = "tfidf"

    def __init__(self):
        self.vec = TfidfVectorizer(
            stop_words="english", sublinear_tf=True, min_df=2, max_df=0.5
        )

    def fit(self, texts):
        self.vec.fit(texts)
        return self

    def encode(self, texts):
        return self.vec.transform(texts)  # already L2-normalised by sklearn


class GloveEmbedder(Embedder):
    """Average of pretrained GloVe word vectors, weighted by IDF.

    Idea: words that appear in few documents (e.g. "orbit") say more about a text
    than words that appear everywhere (e.g. "think"), so they get a bigger weight.
    """

    name = "glove"

    def __init__(self, glove_path: Path = data.RAW_GLOVE, max_words: int = 200_000):
        self.glove_path = glove_path
        self.max_words = max_words
        self.counter = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z]{2,}\b")
        self._vocab: dict[str, int] | None = None
        self._vectors: np.ndarray | None = None
        self.idf: dict[str, float] = {}
        self.default_idf = 1.0

    # -- GloVe loading (cached in memory; file is "word v1 v2 ... v100" per line) --
    def _load_glove(self):
        if self._vectors is not None:
            return
        words, rows = [], []
        with gzip.open(self.glove_path, "rt", encoding="utf-8") as f:
            next(f)  # header line: "<n_words> <dim>"
            for i, line in enumerate(f):
                if i >= self.max_words:
                    break
                parts = line.rstrip().split(" ")
                words.append(parts[0])
                rows.append(np.asarray(parts[1:], dtype=np.float32))
        self._vocab = {w: i for i, w in enumerate(words)}
        self._vectors = np.vstack(rows)

    def fit(self, texts):
        # Document frequency of each word in OUR corpus -> IDF weight.
        X = self.counter.fit_transform(texts)
        df = np.asarray((X > 0).sum(axis=0)).ravel()
        n = X.shape[0]
        idf = np.log((1 + n) / (1 + df)) + 1.0
        self.idf = dict(zip(self.counter.get_feature_names_out(), idf))
        self.default_idf = float(idf.max())  # unseen word = treated as rare
        return self

    def encode(self, texts):
        self._load_glove()
        analyzer = self.counter.build_analyzer()
        out = np.zeros((len(texts), self._vectors.shape[1]), dtype=np.float32)
        for i, text in enumerate(texts):
            acc, total = np.zeros(self._vectors.shape[1], dtype=np.float32), 0.0
            for tok in analyzer(text):
                j = self._vocab.get(tok)
                if j is None:
                    continue
                w = self.idf.get(tok, self.default_idf)
                acc += w * self._vectors[j]
                total += w
            if total > 0:
                out[i] = acc / total
        return normalize(out)

    # Do not pickle the 200k x 100 matrix; it is reloaded from the GloVe file.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_vocab"] = None
        state["_vectors"] = None
        return state


class MiniLMEmbedder(Embedder):
    """Transformer sentence embeddings (384-d). Optional: pip install sentence-transformers."""

    name = "minilm"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _get(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy import

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts):
        return self._get().encode(
            texts, batch_size=64, normalize_embeddings=True, show_progress_bar=len(texts) > 200
        ).astype(np.float32)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        return state


EMBEDDERS = {"tfidf": TfidfEmbedder, "glove": GloveEmbedder, "minilm": MiniLMEmbedder}


def make_embedder(name: str) -> Embedder:
    if name not in EMBEDDERS:
        raise ValueError(f"Unknown embedder '{name}'. Choose from {list(EMBEDDERS)}")
    return EMBEDDERS[name]()


def is_sparse(x) -> bool:
    return sparse.issparse(x)
