"""Flask web UI: indexes a docs folder, search box, cited passages. RTL-aware via dir="auto"."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

from .answer import answer_question
from .normalize import is_arabic
from .retrieval import METHODS, HybridIndex, build_index

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Knowledge Assistant</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --ink:#1c2430; --muted:#5d6b7c; --accent:#1f6f5c; --line:#dde2e8; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#12171d; --card:#1a2129; --ink:#e6ebf0; --muted:#9aa8b8; --accent:#4fb89c; --line:#2b3540; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font-family: system-ui, "Segoe UI", "Noto Naskh Arabic", "Noto Sans Arabic", Tahoma, sans-serif; line-height:1.6; }
  main { max-width: 820px; margin: 0 auto; padding: 24px 16px 48px; }
  h1 { font-size: 1.5rem; margin: 0 0 4px; }
  .sub { color: var(--muted); margin: 0 0 20px; font-size: .95rem; }
  form { display:flex; gap:8px; flex-wrap:wrap; }
  input[type=search] { flex:1 1 280px; padding:10px 12px; font-size:1rem; border:1px solid var(--line);
         border-radius:8px; background:var(--card); color:var(--ink); }
  select, button { padding:10px 14px; font-size:1rem; border-radius:8px; border:1px solid var(--line);
         background:var(--card); color:var(--ink); }
  button { background:var(--accent); color:#fff; border-color:var(--accent); cursor:pointer; }
  .answer, .passage { background:var(--card); border:1px solid var(--line); border-radius:10px;
         padding:14px 16px; margin-top:14px; }
  .answer { border-inline-start: 4px solid var(--accent); }
  .label { font-size:.8rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
  .cite { font-size:.85rem; color:var(--muted); display:flex; gap:10px; flex-wrap:wrap; margin-bottom:6px; }
  .cite b { color:var(--accent); }
  .text { margin:0; unicode-bidi: plaintext; }
  .empty { color:var(--muted); margin-top:18px; }
  footer { margin-top:28px; color:var(--muted); font-size:.85rem; }
</style>
</head>
<body>
<main>
  <h1>Knowledge Assistant</h1>
  <p class="sub">Search {{ n_files }} documents ({{ n_chunks }} passages). Answers are extracted from the files, with citations. No LLM involved.</p>
  <form method="get" action="/">
    <input type="search" name="q" value="{{ q }}" placeholder="Ask in Arabic or English / اكتب سؤالك" dir="auto" autofocus aria-label="Question">
    <select name="method" aria-label="Retrieval method">
      {% for m in methods %}<option value="{{ m }}" {% if m == method %}selected{% endif %}>{{ m }}</option>{% endfor %}
    </select>
    <button type="submit">Search</button>
  </form>
  {% if answer %}
    <div class="answer">
      <div class="label">Answer (extracted)</div>
      <p class="text" dir="auto">{{ answer.text }}</p>
    </div>
    {% for h in answer.hits %}
      <div class="passage">
        <div class="cite">
          <b>[{{ loop.index }}]</b><span>{{ h.chunk.source }}</span><span>{{ h.chunk.locator }}</span>
          <span>score {{ "%.3f"|format(h.score) }}</span>
          {% if h.chunk.section %}<span dir="auto">{{ h.chunk.section }}</span>{% endif %}
        </div>
        <p class="text" dir="auto">{{ h.chunk.text }}</p>
      </div>
    {% endfor %}
  {% elif q %}
    <p class="empty">No result.</p>
  {% endif %}
  <footer>Sources indexed: {{ sources|join(", ") }}</footer>
</main>
</body>
</html>
"""


def create_app(docs_dir: str | Path | None = None, index: HybridIndex | None = None) -> Flask:
    if index is None:
        if docs_dir is None:
            raise ValueError("Provide docs_dir or index")
        index = build_index(docs_dir)
    app = Flask(__name__)
    app.json.ensure_ascii = False
    app.config["INDEX"] = index

    @app.get("/")
    def home():
        q = request.args.get("q", "").strip()
        method = request.args.get("method", "hybrid")
        if method not in METHODS:
            method = "hybrid"
        ans = answer_question(index, q, method=method) if q else None
        return render_template_string(
            PAGE, q=q, method=method, methods=METHODS, answer=ans, n_files=len(index.sources),
            n_chunks=len(index.chunks), sources=index.sources,
            rtl=is_arabic(q) if q else False,
        )

    @app.get("/api/ask")
    def api_ask():
        q = request.args.get("q", "").strip()
        if not q:
            return jsonify({"error": "missing query parameter q"}), 400
        method = request.args.get("method", "hybrid")
        if method not in METHODS:
            return jsonify({"error": f"method must be one of {list(METHODS)}"}), 400
        k = min(max(request.args.get("k", 3, type=int), 1), 10)
        return jsonify(answer_question(index, q, k=k, method=method).to_dict())

    return app
