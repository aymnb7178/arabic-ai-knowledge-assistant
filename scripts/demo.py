"""One-command demo: index sample docs -> ask questions -> evaluate -> charts. No network, no LLM."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from knowledge_assistant.answer import answer_question  # noqa: E402
from knowledge_assistant.charts import plot_method_comparison, plot_rank_distribution  # noqa: E402
from knowledge_assistant.evaluate import evaluate, load_questions  # noqa: E402
from knowledge_assistant.retrieval import build_index, save_index  # noqa: E402

DEMO_QUESTIONS = [
    "What is the 3-2-1 backup rule?",
    "ما المشاريع العملاقة المرتبطة بالرؤية؟",
    "How do I stop pandas from duplicating rows after a merge?",
    "How do I cook kabsa?",
]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    docs, out = ROOT / "docs", ROOT / "output"
    index = build_index(ROOT / "data" / "sample_docs")
    save_index(index, out / "index.joblib")
    print(f"Indexed {len(index.sources)} files into {len(index.chunks)} chunks")

    for q in DEMO_QUESTIONS:
        ans = answer_question(index, q, k=3)
        print(f"\nQ: {q}\nA: {ans.text}")
        for c in ans.citations:
            print(f"   [{c.n}] {c.source} | {c.locator} | score {c.score:.3f}")

    result = evaluate(index, load_questions(ROOT / "data" / "eval_questions.json"))
    print(f"\nEvaluation: {result['n_questions']} questions, {result['n_chunks']} chunks")
    print(f"{'method':<12} {'src@1':>6} {'src@3':>6} {'psg@1':>6} {'psg@3':>6}")
    for m, r in result["methods"].items():
        print(f"{m:<12} {r['source_hit@1']:>6.2f} {r['source_hit@3']:>6.2f} "
              f"{r['passage_hit@1']:>6.2f} {r['passage_hit@3']:>6.2f}")

    docs.mkdir(exist_ok=True)
    (docs / "eval_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nWrote", plot_method_comparison(result, docs / "retrieval_eval.png").relative_to(ROOT))
    print("Wrote", plot_rank_distribution(result, docs / "retrieval_ranks.png").relative_to(ROOT))
    print("Wrote", (docs / "eval_results.json").relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
