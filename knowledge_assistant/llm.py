"""Optional, pluggable LLM layer. The project works without any LLM (see ExtractiveBackend)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .retrieval import Hit

PROMPT_TEMPLATE = (
    "Answer the question using ONLY the numbered passages. Cite passages like [1]. "
    "If the passages do not contain the answer, say you do not know.\n\n"
    "{passages}\n\nQuestion: {question}\nAnswer:"
)


def build_prompt(question: str, hits: list[Hit]) -> str:
    passages = "\n\n".join(
        f"[{i}] ({h.chunk.source}, {h.chunk.locator}) {h.chunk.text}" for i, h in enumerate(hits, 1)
    )
    return PROMPT_TEMPLATE.format(passages=passages, question=question)


class LLMBackend(ABC):
    """Implement ``generate`` to let a language model write the final answer from retrieved passages."""

    name = "llm"

    @abstractmethod
    def generate(self, question: str, hits: list[Hit]) -> str:
        """Return answer text. ``hits`` are the retrieved passages, best first."""


class ExtractiveBackend(LLMBackend):
    """Default: no model. Returns the best passage's first sentence(s) verbatim (see answer.py)."""

    name = "extractive"

    def generate(self, question: str, hits: list[Hit]) -> str:
        from .answer import extractive_text

        return extractive_text(question, hits)


class ExampleHTTPBackend(LLMBackend):
    """Documented stub, NOT implemented and never called by tests or the demo.

    To use a hosted or local model, subclass :class:`LLMBackend`, build the prompt with
    :func:`build_prompt`, send it to your provider with your own client and API key (read the key
    from an environment variable, never commit it) and return the text of the reply.
    """

    name = "example-http (stub)"

    def __init__(self, endpoint: str = "", api_key_env: str = "LLM_API_KEY"):
        self.endpoint, self.api_key_env = endpoint, api_key_env

    def generate(self, question: str, hits: list[Hit]) -> str:
        raise NotImplementedError(
            "ExampleHTTPBackend is a stub. Subclass LLMBackend and call your provider here."
        )
