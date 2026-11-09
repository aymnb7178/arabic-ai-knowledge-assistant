"""Shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from knowledge_assistant.retrieval import build_index  # noqa: E402

SAMPLE_DOCS = ROOT / "data" / "sample_docs"


@pytest.fixture(scope="session")
def index():
    return build_index(SAMPLE_DOCS)
