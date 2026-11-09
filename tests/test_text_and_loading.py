import zipfile

import pytest

from knowledge_assistant.chunking import chunk_document, split_sentences
from knowledge_assistant.loaders import Document, Segment, load_docx, load_document, load_folder, load_md, load_txt
from knowledge_assistant.normalize import is_arabic, normalize_text, tokenize


def test_normalize_removes_tashkeel_and_tatweel():
    assert normalize_text("كَتَبَ") == "كتب"
    assert normalize_text("مــــرحبا") == "مرحبا"


def test_normalize_letter_variants():
    assert normalize_text("أحمد إبراهيم آمن") == "احمد ابراهيم امن"
    assert normalize_text("مدينة") == normalize_text("مدينه")  # taa marbuta -> haa
    assert normalize_text("على") == normalize_text("علي")  # alef maksura -> yaa


def test_normalize_arabic_indic_digits_and_case():
    assert normalize_text("رؤية ٢٠٣٠ و ۲۰۳۰") == "رويه 2030 و 2030"
    assert normalize_text("Hello  WORLD") == "hello world"


def test_tokenize_drops_stopwords_and_stems():
    assert "the" not in tokenize("the passwords")
    assert tokenize("Passwords") == tokenize("password")
    assert tokenize("المشاريع") == tokenize("مشاريع")
    assert tokenize("في المدينة") == tokenize("المدينه")


def test_is_arabic():
    assert is_arabic("مرحبا بالعالم")
    assert not is_arabic("hello world")


def test_split_sentences_arabic_and_english():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert len(split_sentences("ما هذا؟ هذا اختبار. نعم")) == 3


def test_load_txt_and_md(tmp_path):
    t = tmp_path / "a.txt"
    t.write_text("Title line\n\nFirst para.\n\nSecond para.", encoding="utf-8")
    d = load_txt(t)
    assert [s.para for s in d.segments] == [1, 2, 3]
    assert d.segments[0].is_heading
    m = tmp_path / "b.md"
    m.write_text("# Head\n\nBody text here.", encoding="utf-8")
    md = load_md(m)
    assert md.segments[0].is_heading and md.segments[0].text == "Head"


def _make_docx(path, paragraphs):
    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    body = ""
    for text, style in paragraphs:
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        body += f"<w:p>{ppr}<w:r><w:t>{text}</w:t></w:r></w:p>"
    xml = f'<?xml version="1.0" encoding="UTF-8"?><w:document {ns}><w:body>{body}</w:body></w:document>'
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", xml)


def test_load_docx_text_and_headings(tmp_path):
    p = tmp_path / "x.docx"
    _make_docx(p, [("Heading A", "Heading1"), ("مرحبا بالعالم", None), ("", None), ("Second", None)])
    d = load_docx(p)
    assert [s.text for s in d.segments] == ["Heading A", "مرحبا بالعالم", "Second"]
    assert d.segments[0].is_heading and not d.segments[1].is_heading
    assert d.segments[2].para == 3


def test_load_folder_skips_unsupported_and_sets_relative_names(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "n.txt").write_text("Title\n\nBody.", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    docs = load_folder(tmp_path)
    assert [d.source for d in docs] == ["sub/n.txt"]
    with pytest.raises(ValueError):
        load_document(tmp_path / "image.png")


def test_chunking_overlap_locators_and_size():
    sents = [f"Sentence number {i} has some filler words in it." for i in range(1, 13)]
    doc = Document("f.txt", [Segment(" ".join(sents[:6]), 1), Segment(" ".join(sents[6:]), 2)])
    chunks = chunk_document(doc, max_chars=150, overlap_sentences=1)
    assert len(chunks) > 2
    assert all(len(c.text) <= 150 + 60 for c in chunks)
    # the last sentence of a chunk reappears at the start of the next one
    assert split_sentences(chunks[0].text)[-1] == split_sentences(chunks[1].text)[0]
    assert chunks[0].locator == "paragraph 1"
    assert any(c.locator == "paragraphs 1-2" for c in chunks)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))


def test_chunking_headings_start_new_sections_without_overlap():
    doc = Document("f.md", [
        Segment("A", 1, is_heading=True), Segment("Alpha one. Alpha two.", 2),
        Segment("B", 3, is_heading=True), Segment("Beta one. Beta two.", 4),
    ])
    chunks = chunk_document(doc, max_chars=500)
    assert [c.section for c in chunks] == ["A", "B"]
    assert "Alpha" not in chunks[1].text


def test_chunk_page_locator_for_pdf_like_segments():
    doc = Document("p.pdf", [Segment("Only text here on page.", 1, page=3)])
    assert chunk_document(doc)[0].locator == "page 3"
