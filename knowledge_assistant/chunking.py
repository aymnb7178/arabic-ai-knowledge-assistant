"""Sentence-aware chunking with sentence overlap; chunks remember file and page/paragraph."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .loaders import Document

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟…])\s+|\n+")


@dataclass
class Chunk:
    chunk_id: int
    source: str
    locator: str  # e.g. "paragraph 4", "paragraphs 4-5", "page 2"
    text: str
    section: str = ""

    @property
    def index_text(self) -> str:
        """Text that is indexed: the section heading helps short passages match their topic."""
        return f"{self.section}. {self.text}" if self.section else self.text


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _locator(units: list[tuple[str, int, int | None]]) -> str:
    pages = [u[2] for u in units if u[2] is not None]
    if pages:
        lo, hi = min(pages), max(pages)
        return f"page {lo}" if lo == hi else f"pages {lo}-{hi}"
    paras = [u[1] for u in units]
    lo, hi = min(paras), max(paras)
    return f"paragraph {lo}" if lo == hi else f"paragraphs {lo}-{hi}"


def chunk_document(
    doc: Document, max_chars: int = 450, overlap_sentences: int = 1, start_id: int = 0
) -> list[Chunk]:
    """Greedily pack whole sentences up to ``max_chars``; the last ``overlap_sentences`` sentences of a
    chunk are repeated at the start of the next one. A heading starts a new section and never overlaps."""
    if max_chars < 50:
        raise ValueError("max_chars must be at least 50")
    chunks: list[Chunk] = []
    section = ""
    cur: list[tuple[str, int, int | None]] = []  # (sentence, para, page)
    fresh = 0  # sentences in cur that were not carried over as overlap

    def flush() -> None:
        nonlocal cur, fresh
        if fresh:
            chunks.append(
                Chunk(start_id + len(chunks), doc.source, _locator(cur), " ".join(u[0] for u in cur), section)
            )
        cur, fresh = [], 0

    for seg in doc.segments:
        if seg.is_heading:
            flush()
            section = seg.text
            continue
        for sent in split_sentences(seg.text):
            size = sum(len(u[0]) + 1 for u in cur)
            if fresh and size + len(sent) > max_chars:
                tail = cur[-overlap_sentences:] if overlap_sentences > 0 else []
                flush()
                cur = list(tail)
            cur.append((sent, seg.para, seg.page))
            fresh += 1
    flush()
    return chunks


def chunk_documents(docs: list[Document], max_chars: int = 450, overlap_sentences: int = 1) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, max_chars, overlap_sentences, start_id=len(out)))
    return out
