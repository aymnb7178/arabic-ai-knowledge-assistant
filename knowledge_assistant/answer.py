"""Extractive answers: the answer text is copied from retrieved passages, never generated."""

from __future__ import annotations

from dataclasses import dataclass, field

from .chunking import split_sentences
from .llm import ExtractiveBackend, LLMBackend
from .normalize import tokenize
from .retrieval import Hit, HybridIndex

NO_ANSWER = "No sufficiently relevant passage was found in the indexed documents."


@dataclass
class Citation:
    n: int
    source: str
    locator: str
    score: float


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    mode: str = "extractive"

    @property
    def answered(self) -> bool:
        return bool(self.hits) and self.text != NO_ANSWER

    @property
    def citations(self) -> list[Citation]:
        return [Citation(i, h.chunk.source, h.chunk.locator, round(h.score, 3)) for i, h in enumerate(self.hits, 1)]

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "mode": self.mode,
            "passages": [
                {
                    "n": i,
                    "source": h.chunk.source,
                    "locator": h.chunk.locator,
                    "section": h.chunk.section,
                    "score": round(h.score, 4),
                    "text": h.chunk.text,
                }
                for i, h in enumerate(self.hits, 1)
            ],
        }


def extractive_text(question: str, hits: list[Hit], max_sentences: int = 2) -> str:
    """Pick the sentences of the top passage that cover most query terms; keep original order; cite [1]."""
    if not hits:
        return NO_ANSWER
    q_tokens = set(tokenize(question))
    sentences = split_sentences(hits[0].chunk.text)
    scored = [
        (len(q_tokens & set(tokenize(s))), -i, s) for i, s in enumerate(sentences)
    ]
    best = sorted(scored, reverse=True)[:max_sentences]
    chosen = [s for _, neg_i, s in sorted(best, key=lambda t: -t[1])]
    return " ".join(chosen) + " [1]"


def answer_question(
    index: HybridIndex,
    question: str,
    k: int = 3,
    method: str = "hybrid",
    min_score: float = 0.12,
    llm: LLMBackend | None = None,
) -> Answer:
    """Retrieve the top ``k`` passages and build an answer. Below ``min_score`` the top hit is discarded."""
    hits = index.search(question, k=k, method=method)
    if method == "hybrid":  # the threshold is only meaningful on the 0-1 hybrid scale
        hits = [h for h in hits if h.score >= min_score]
    backend = llm or ExtractiveBackend()
    if not hits:
        return Answer(question, NO_ANSWER, [], backend.name)
    return Answer(question, backend.generate(question, hits), hits, backend.name)
