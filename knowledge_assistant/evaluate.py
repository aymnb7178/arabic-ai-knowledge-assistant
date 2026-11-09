"""Retrieval evaluation on a small hand-written question set."""

from __future__ import annotations

import json
from pathlib import Path

from .normalize import normalize_text
from .retrieval import METHODS, HybridIndex


def load_questions(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate(index: HybridIndex, questions: list[dict], ks: tuple[int, ...] = (1, 3)) -> dict:
    """For each method compute hit@k by source file and by passage (expected snippet inside a top-k chunk)."""
    kmax = max(ks)
    result: dict = {"n_questions": len(questions), "n_chunks": len(index.chunks), "methods": {}}
    for method in METHODS:
        src_hits = {k: 0 for k in ks}
        snip_hits = {k: 0 for k in ks}
        rr_sum = 0.0
        per_q = []
        for q in questions:
            hits = index.search(q["question"], k=kmax, method=method)
            want = normalize_text(q["expected_snippet"])
            src_rank = next((i for i, h in enumerate(hits, 1) if h.chunk.source == q["expected_source"]), None)
            snip_rank = next(
                (i for i, h in enumerate(hits, 1)
                 if h.chunk.source == q["expected_source"] and want in normalize_text(h.chunk.text)),
                None,
            )
            for k in ks:
                src_hits[k] += src_rank is not None and src_rank <= k
                snip_hits[k] += snip_rank is not None and snip_rank <= k
            rr_sum += 1 / snip_rank if snip_rank else 0.0
            per_q.append({"question": q["question"], "lang": q["lang"], "source_rank": src_rank,
                          "passage_rank": snip_rank})
        n = len(questions)
        result["methods"][method] = {
            **{f"source_hit@{k}": src_hits[k] / n for k in ks},
            **{f"passage_hit@{k}": snip_hits[k] / n for k in ks},
            "passage_mrr@%d" % kmax: rr_sum / n,
            "per_question": per_q,
        }
    return result
