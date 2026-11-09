"""Document loaders: .txt, .md, .docx (zipfile + XML) and .pdf (only if pypdf is installed)."""

from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional dependency
    import pypdf  # type: ignore
except ImportError:  # pragma: no cover - depends on environment
    pypdf = None

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
SUPPORTED = {".txt", ".md", ".docx"} | ({".pdf"} if pypdf is not None else set())


@dataclass
class Segment:
    """One paragraph-like block. ``para`` is a 1-based count of non-empty blocks (headings included)."""

    text: str
    para: int
    page: int | None = None
    is_heading: bool = False


@dataclass
class Document:
    source: str  # file name used in citations
    segments: list[Segment] = field(default_factory=list)


def pdf_supported() -> bool:
    return pypdf is not None


def _split_blocks(text: str) -> list[str]:
    return [b.strip() for b in re.split(r"\n\s*\n", text.replace("\r\n", "\n")) if b.strip()]


def _from_text_blocks(source: str, text: str, markdown: bool) -> Document:
    doc = Document(source)
    for i, block in enumerate(_split_blocks(text), start=1):
        heading = False
        if markdown and re.match(r"#{1,6}\s", block):
            block, heading = re.sub(r"^#{1,6}\s+", "", block).strip(), True
        elif not markdown and i == 1 and "\n" not in block and len(block) < 100:
            heading = True  # plain-text files: a short first line is treated as the title
        doc.segments.append(Segment(block, para=i, is_heading=heading))
    return doc


def load_txt(path: Path) -> Document:
    return _from_text_blocks(path.name, path.read_text(encoding="utf-8-sig"), markdown=False)


def load_md(path: Path) -> Document:
    return _from_text_blocks(path.name, path.read_text(encoding="utf-8-sig"), markdown=True)


def load_docx(path: Path) -> Document:
    """Read paragraphs from word/document.xml. Headings are detected from the paragraph style name."""
    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))
    doc = Document(path.name)
    n = 0
    for p in root.iter(f"{W_NS}p"):
        text = "".join(t.text or "" for t in p.iter(f"{W_NS}t")).strip()
        if not text:
            continue
        n += 1
        style = p.find(f"{W_NS}pPr/{W_NS}pStyle")
        style_name = style.get(f"{W_NS}val", "") if style is not None else ""
        doc.segments.append(
            Segment(text, para=n, is_heading=style_name.lower().startswith(("heading", "title")))
        )
    return doc


def load_pdf(path: Path) -> Document:
    if pypdf is None:
        raise RuntimeError("PDF support needs the optional 'pypdf' package (pip install pypdf).")
    reader = pypdf.PdfReader(str(path))
    doc = Document(path.name)
    n = 0
    for page_no, page in enumerate(reader.pages, start=1):
        for block in _split_blocks(page.extract_text() or ""):
            n += 1
            doc.segments.append(Segment(block, para=n, page=page_no))
    return doc


_LOADERS = {".txt": load_txt, ".md": load_md, ".docx": load_docx, ".pdf": load_pdf}


def load_document(path: str | Path) -> Document:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in _LOADERS:
        raise ValueError(f"Unsupported file type: {ext}")
    if ext not in SUPPORTED:
        raise RuntimeError("PDF support needs the optional 'pypdf' package (pip install pypdf).")
    return _LOADERS[ext](path)


def load_folder(folder: str | Path) -> list[Document]:
    """Load every supported file under ``folder`` (sorted, recursive). Unsupported files are skipped."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a folder: {folder}")
    docs = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            doc = load_document(path)
            doc.source = path.relative_to(folder).as_posix()
            if doc.segments:
                docs.append(doc)
    return docs
