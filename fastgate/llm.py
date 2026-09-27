"""Optional LLM pieces via LangChain: grounded answer generation and an LLM-router baseline.

Enabled only when FASTGATE_LLM is set (e.g. "anthropic:claude-haiku-4-5-20251001")
and the matching langchain provider package is installed.
"""
import time
from typing import Literal

from pydantic import BaseModel, Field

from . import config as C

LANG_NAMES = {"en": "English", "uz": "Uzbek", "ru": "Russian", "mixed": "the student's language"}


def available() -> bool:
    if not C.LLM:
        return False
    try:
        import langchain  # noqa: F401
        return True
    except ImportError:
        return False


def _model():
    from langchain.chat_models import init_chat_model
    return init_chat_model(C.LLM, temperature=0)


class IntentOut(BaseModel):
    intent: Literal[tuple(C.INTENTS)]  # type: ignore[valid-type]
    confidence: float = Field(ge=0, le=1, description="Probability that the chosen intent is correct")


async def llm_route(query: str) -> tuple[str, float, float]:
    """Baseline: the same intent decision made by an LLM with structured output."""
    options = "\n".join(f"- {k}: {v}" for k, v in C.INTENTS.items())
    prompt = (f"Classify the university student's message into exactly one intent.\n{options}\n\n"
              f"Message: {query}\nAlso give the probability that your answer is correct.")
    t0 = time.perf_counter()
    out = await _model().with_structured_output(IntentOut).ainvoke(prompt)
    return out.intent, float(out.confidence), (time.perf_counter() - t0) * 1000


async def answer(query: str, language: str, passages: list[dict]) -> str:
    context = "\n\n".join(f"[{p['id']}] {p['text']}" for p in passages)
    prompt = (
        "You are a university helpdesk assistant. Answer ONLY from the sources below. "
        "If they do not contain the answer, say so. Cite source ids in square brackets. "
        f"Reply in {LANG_NAMES.get(language, 'the student’s language')}, in at most 4 sentences.\n\n"
        f"Sources:\n{context}\n\nStudent: {query}"
    )
    out = await _model().ainvoke(prompt)
    return out.content if isinstance(out.content, str) else str(out.content)
