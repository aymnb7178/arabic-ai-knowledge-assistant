"""Command line interface: index, ask, eval, serve."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .answer import answer_question
from .evaluate import evaluate, load_questions
from .retrieval import METHODS, HybridIndex, build_index, load_index, save_index

DEFAULT_INDEX = "output/index.joblib"


def _get_index(args) -> HybridIndex:
    if args.docs:
        return build_index(args.docs)
    if Path(args.index).exists():
        return load_index(args.index)
    raise SystemExit(f"No index at {args.index}. Run `index <docs_dir>` first or pass --docs.")


def cmd_index(args) -> int:
    idx = build_index(args.docs_dir, max_chars=args.max_chars, overlap_sentences=args.overlap)
    save_index(idx, args.out)
    print(f"Indexed {len(idx.sources)} files into {len(idx.chunks)} chunks -> {args.out}")
    for s in idx.sources:
        print(f"  {s}: {sum(c.source == s for c in idx.chunks)} chunks")
    return 0


def cmd_ask(args) -> int:
    idx = _get_index(args)
    ans = answer_question(idx, args.question, k=args.k, method=args.method)
    if args.json:
        print(json.dumps(ans.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(f"Q: {ans.question}\n")
    print(f"Answer: {ans.text}\n")
    for c, h in zip(ans.citations, ans.hits):
        print(f"[{c.n}] {c.source} | {c.locator} | score {c.score:.3f}")
        print(f"    {h.chunk.text}")
    return 0


def cmd_eval(args) -> int:
    idx = build_index(args.docs)
    res = evaluate(idx, load_questions(args.questions))
    print(f"{res['n_questions']} questions, {res['n_chunks']} chunks")
    print(f"{'method':<12} {'src@1':>6} {'src@3':>6} {'psg@1':>6} {'psg@3':>6}")
    for m, r in res["methods"].items():
        print(f"{m:<12} {r['source_hit@1']:>6.2f} {r['source_hit@3']:>6.2f} "
              f"{r['passage_hit@1']:>6.2f} {r['passage_hit@3']:>6.2f}")
    return 0


def cmd_serve(args) -> int:
    from .web import create_app

    create_app(docs_dir=args.docs).run(host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="knowledge-assistant", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("index", help="build and save an index from a docs folder")
    a.add_argument("docs_dir")
    a.add_argument("--out", default=DEFAULT_INDEX)
    a.add_argument("--max-chars", type=int, default=450)
    a.add_argument("--overlap", type=int, default=1, help="sentences repeated between chunks")
    a.set_defaults(func=cmd_index)

    a = sub.add_parser("ask", help="ask a question (Arabic or English)")
    a.add_argument("question")
    a.add_argument("--index", default=DEFAULT_INDEX)
    a.add_argument("--docs", help="build the index on the fly from this folder instead of loading one")
    a.add_argument("-k", type=int, default=3)
    a.add_argument("--method", choices=METHODS, default="hybrid")
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_ask)

    a = sub.add_parser("eval", help="run the retrieval evaluation")
    a.add_argument("--docs", default="data/sample_docs")
    a.add_argument("--questions", default="data/eval_questions.json")
    a.set_defaults(func=cmd_eval)

    a = sub.add_parser("serve", help="start the Flask web UI")
    a.add_argument("--docs", default="data/sample_docs")
    a.add_argument("--host", default="127.0.0.1")
    a.add_argument("--port", type=int, default=5000)
    a.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
