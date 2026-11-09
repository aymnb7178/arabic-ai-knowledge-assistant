import math
import subprocess
import sys

import pytest

from knowledge_assistant import cli
from knowledge_assistant.answer import NO_ANSWER, answer_question
from knowledge_assistant.evaluate import evaluate, load_questions
from knowledge_assistant.llm import ExampleHTTPBackend, LLMBackend, build_prompt
from knowledge_assistant.retrieval import BM25, HybridIndex, load_index, save_index
from knowledge_assistant.web import create_app
from tests.conftest import ROOT, SAMPLE_DOCS


def test_bm25_matches_hand_computation():
    bm = BM25(k1=1.5, b=0.75).fit([["a", "b"], ["b", "c", "c"]])
    idf_a = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    expected = idf_a * 1 * 2.5 / (1 + 1.5 * (1 - 0.75 + 0.75 * 2 / 2.5))
    assert bm.score(["a"])[0] == pytest.approx(expected)
    assert bm.score(["a"])[1] == 0
    assert bm.score(["zzz"]).sum() == 0


def test_bm25_prefers_higher_term_frequency():
    bm = BM25().fit([["x", "y", "y"], ["x", "z", "w"], ["q", "r", "s"]])
    s = bm.score(["y"])
    assert s[0] > s[1] and s[2] == 0


def test_english_query_finds_right_file(index):
    hit = index.search("What is the 3-2-1 backup rule?", k=1)[0]
    assert hit.chunk.source == "cybersecurity_basics_en.md"


def test_arabic_query_finds_arabic_file_even_with_diacritics(index):
    plain = index.search("ما هي قاعدة النسخ الاحتياطي؟", k=1)[0]
    marked = index.search("مَا هِيَ قَاعِدَةُ النُّسَخِ الِاحْتِيَاطِيِّ؟", k=1)[0]
    assert plain.chunk.source == "cybersecurity_basics_ar.md"
    assert marked.chunk.chunk_id == plain.chunk.chunk_id


def test_arabic_indic_digits_in_query(index):
    hit = index.search("قاعدة ٣-٢-١ للنسخ الاحتياطي", k=1)[0]
    assert hit.chunk.source == "cybersecurity_basics_ar.md"


def test_hybrid_score_is_weighted_sum_of_components(index):
    q = "What is phishing?"
    top = index.search(q, k=1)[0]
    comp = top.components
    bm = index.bm25.score(__import__("knowledge_assistant.normalize", fromlist=["tokenize"]).tokenize(q))
    expected = (0.4 * comp["bm25"] / bm.max() + 0.3 * comp["tfidf_word"] + 0.3 * comp["tfidf_char"])
    assert top.score == pytest.approx(expected)


def test_all_methods_return_sorted_scores(index):
    for m in ("hybrid", "bm25", "tfidf_word", "tfidf_char"):
        scores = [h.score for h in index.search("software updates", k=5, method=m)]
        assert scores == sorted(scores, reverse=True) and scores
    with pytest.raises(ValueError):
        index.search("x", method="nope")


def test_answer_has_citations_with_file_locator_score(index):
    ans = answer_question(index, "What is the role of the Public Investment Fund?")
    assert ans.answered and ans.text.endswith("[1]")
    c = ans.citations[0]
    assert c.source == "vision2030_facts_en.txt" and c.locator.startswith("paragraph") and c.score > 0
    assert ans.to_dict()["passages"][0]["source"] == c.source


def test_answer_text_is_copied_from_passage(index):
    ans = answer_question(index, "When was Saudi Vision 2030 announced?")
    body = ans.text[: -len(" [1]")]
    assert "April 2016" in body and body in " ".join(ans.hits[0].chunk.text.split())


def test_out_of_domain_question_gets_no_answer(index):
    ans = answer_question(index, "How do I cook kabsa?")
    assert ans.text == NO_ANSWER and not ans.answered


def test_llm_interface_is_abstract_and_pluggable(index):
    with pytest.raises(TypeError):
        LLMBackend()

    class Fake(LLMBackend):
        name = "fake"

        def generate(self, question, hits):
            return "FAKE:" + build_prompt(question, hits)[:20]

    ans = answer_question(index, "What is phishing?", llm=Fake())
    assert ans.text.startswith("FAKE:") and ans.mode == "fake"
    with pytest.raises(NotImplementedError):
        ExampleHTTPBackend().generate("q", [])


def test_prompt_contains_numbered_passages_with_citations(index):
    hits = index.search("What is phishing?", k=2)
    p = build_prompt("What is phishing?", hits)
    assert "[1] (" in p and "[2] (" in p and "Question: What is phishing?" in p


def test_save_and_load_index_roundtrip(index, tmp_path):
    save_index(index, tmp_path / "i.joblib")
    again = load_index(tmp_path / "i.joblib")
    q = "ما هي محاور الرؤية الثلاثة؟"
    assert [h.chunk.chunk_id for h in again.search(q)] == [h.chunk.chunk_id for h in index.search(q)]


def test_empty_index_rejected():
    with pytest.raises(ValueError):
        HybridIndex([])


def test_cli_index_and_ask(tmp_path, capsys):
    out = tmp_path / "idx.joblib"
    assert cli.main(["index", str(SAMPLE_DOCS), "--out", str(out)]) == 0
    assert "chunks" in capsys.readouterr().out
    assert cli.main(["ask", "What is phishing?", "--index", str(out)]) == 0
    text = capsys.readouterr().out
    assert "cybersecurity_basics_en.md" in text and "score" in text
    assert cli.main(["ask", "ما هو التصيد الاحتيالي؟", "--index", str(out), "--json"]) == 0
    assert '"source": "cybersecurity_basics_ar.md"' in capsys.readouterr().out


def test_cli_module_entrypoint_runs():
    r = subprocess.run([sys.executable, "-m", "knowledge_assistant", "ask", "least privilege",
                        "--docs", str(SAMPLE_DOCS)], capture_output=True, text=True, cwd=ROOT, encoding="utf-8")
    assert r.returncode == 0 and "least privilege" in r.stdout.lower()


def test_web_ui_and_api(index):
    client = create_app(index=index).test_client()
    page = client.get("/")
    assert page.status_code == 200 and 'dir="auto"' in page.get_data(as_text=True)
    res = client.get("/", query_string={"q": "ما هو مبدأ أقل الصلاحيات؟"})
    html = res.get_data(as_text=True)
    assert "cybersecurity_basics_ar.md" in html and "الوصول اللازم" in html
    api = client.get("/api/ask", query_string={"q": "What is phishing?", "k": 2}).get_json()
    assert len(api["passages"]) == 2 and api["passages"][0]["source"] == "cybersecurity_basics_en.md"
    assert client.get("/api/ask").status_code == 400


def test_web_escapes_html_in_query(index):
    html = create_app(index=index).test_client().get("/", query_string={"q": "<script>alert(1)</script>"}).get_data(as_text=True)
    assert "<script>alert(1)</script>" not in html


def test_evaluation_runs_and_is_reasonable(index):
    qs = load_questions(ROOT / "data" / "eval_questions.json")
    assert len(qs) == 20 and {q["expected_source"] for q in qs} <= set(index.sources)
    res = evaluate(index, qs)
    h = res["methods"]["hybrid"]
    assert 0 <= h["source_hit@1"] <= h["source_hit@3"] <= 1
    assert h["source_hit@3"] >= 0.8  # sanity floor on a tiny corpus, not a quality claim


def test_charts_are_written(index, tmp_path):
    from knowledge_assistant.charts import plot_method_comparison, plot_rank_distribution

    res = evaluate(index, load_questions(ROOT / "data" / "eval_questions.json"))
    for p in (plot_method_comparison(res, tmp_path / "a.png"), plot_rank_distribution(res, tmp_path / "b.png")):
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
