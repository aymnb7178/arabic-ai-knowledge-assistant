"""Retrieval: word TF-IDF, character n-gram TF-IDF, hand-written BM25 and a hybrid of the three."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunking import Chunk, chunk_documents
from .loaders import load_folder
from .normalize import normalize_text, tokenize

METHODS = ("hybrid", "bm25", "tfidf_word", "tfidf_char")
DEFAULT_WEIGHTS = {"bm25": 0.4, "tfidf_word": 0.3, "tfidf_char": 0.3}


class BM25:
    """Okapi BM25 written from scratch (idf = ln(1 + (N - df + 0.5) / (df + 0.5)))."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n_docs = 0
        self.avgdl = 0.0
        self.doc_len: list[int] = []
        self.tf: list[Counter] = []
        self.idf: dict[str, float] = {}

    def fit(self, docs: list[list[str]]) -> "BM25":
        self.n_docs = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / max(self.n_docs, 1)
        self.tf = [Counter(d) for d in docs]
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        self.idf = {t: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        return self

    def score(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs)
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(term, 0)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                    scores[i] += idf * f * (self.k1 + 1) / (f + norm)
        return scores


@dataclass
class Hit:
    chunk: Chunk
    score: float
    components: dict[str, float] = field(default_factory=dict)


class HybridIndex:
    """Holds chunks and three scorers. ``hybrid`` = weighted sum of max-normalised BM25, word TF-IDF
    cosine and char n-gram TF-IDF cosine."""

    def __init__(self, chunks: list[Chunk], weights: dict[str, float] | None = None):
        if not chunks:
            raise ValueError("Cannot build an index from zero chunks.")
        self.chunks = chunks
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        texts = [c.index_text for c in chunks]
        self.word_vec = TfidfVectorizer(analyzer=tokenize, sublinear_tf=True)
        self.word_mat = self.word_vec.fit_transform(texts)
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), preprocessor=normalize_text, sublinear_tf=True
        )
        self.char_mat = self.char_vec.fit_transform(texts)
        self.bm25 = BM25().fit([tokenize(t) for t in texts])

    # -- scoring ---------------------------------------------------------
    def component_scores(self, query: str) -> dict[str, np.ndarray]:
        n = len(self.chunks)
        word = (self.word_mat @ self.word_vec.transform([query]).T).toarray().ravel() if tokenize(query) else np.zeros(n)
        char = (self.char_mat @ self.char_vec.transform([query]).T).toarray().ravel()
        return {"bm25": self.bm25.score(tokenize(query)), "tfidf_word": word, "tfidf_char": char}

    def scores(self, query: str, method: str = "hybrid") -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
        comp = self.component_scores(query)
        if method != "hybrid":
            return comp[method], comp
        bm = comp["bm25"]
        bm_norm = bm / bm.max() if bm.max() > 0 else bm
        total = (
            self.weights["bm25"] * bm_norm
            + self.weights["tfidf_word"] * comp["tfidf_word"]
            + self.weights["tfidf_char"] * comp["tfidf_char"]
        )
        return total, comp

    def search(self, query: str, k: int = 3, method: str = "hybrid") -> list[Hit]:
        total, comp = self.scores(query, method)
        order = np.argsort(-total, kind="stable")[:k]
        return [
            Hit(self.chunks[i], float(total[i]), {m: float(v[i]) for m, v in comp.items()})
            for i in order
            if total[i] > 0
        ]

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks})


def build_index(
    docs_dir: str | Path, max_chars: int = 450, overlap_sentences: int = 1
) -> HybridIndex:
    docs = load_folder(docs_dir)
    return HybridIndex(chunk_documents(docs, max_chars, overlap_sentences))


def save_index(index: HybridIndex, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, path)


def load_index(path: str | Path) -> HybridIndex:
    """Load an index saved by :func:`save_index`. joblib uses pickle: only load files you created."""
    return joblib.load(path)
