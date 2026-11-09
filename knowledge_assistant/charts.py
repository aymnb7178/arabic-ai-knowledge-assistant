"""matplotlib charts for the evaluation (English labels only, so no Arabic font is required)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"hit@1": "#1f6f5c", "hit@3": "#8fc7b5"}


def plot_method_comparison(result: dict, out: str | Path) -> Path:
    methods = list(result["methods"])
    x = range(len(methods))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    w = 0.38
    for off, k in ((-w / 2, 1), (w / 2, 3)):
        vals = [result["methods"][m][f"source_hit@{k}"] for m in methods]
        bars = ax.bar([i + off for i in x], vals, w, label=f"hit@{k}", color=COLORS[f"hit@{k}"])
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], padding=2, fontsize=9)
    ax.set_xticks(list(x), methods)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("share of questions")
    ax.set_title(f"Retrieval hit rate by method ({result['n_questions']} questions, {result['n_chunks']} chunks)")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return Path(out)


def plot_rank_distribution(result: dict, out: str | Path, method: str = "hybrid") -> Path:
    per_q = result["methods"][method]["per_question"]
    labels = ["rank 1", "rank 2", "rank 3", "not in top 3"]
    counts = [0, 0, 0, 0]
    for q in per_q:
        r = q["source_rank"]
        counts[(r - 1) if r else 3] += 1
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(labels, counts, color=["#1f6f5c", "#5fa994", "#8fc7b5", "#c9a227"])
    ax.bar_label(bars, padding=2)
    ax.set_ylabel("questions")
    ax.set_title(f"Where the expected source file first appears ({method})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return Path(out)
