"""Jev decision layer + deterministic routing.

Two Jev calls run in parallel per query:
  1. triage   : state = query only (language, intent, urgency, wants_human)
  2. grounding: state = query + retrieved passages (one Noul per passage)
Splitting them keeps irrelevant passages out of the triage state, following
TypeSafe's guidance that unrelated context lowers accuracy.
"""
import asyncio
import time
from dataclasses import dataclass, field

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul

from . import config as C

_client: AsyncTypeSafeClient | None = None


def get_client() -> AsyncTypeSafeClient:
    global _client
    if _client is None:
        if C.BACKEND == "mock":
            from .mock import mock_transport
            _client = AsyncTypeSafeClient(api_key="mock", model="mock-jev", transport=mock_transport())
        elif C.BACKEND == "cloudflare":
            import os
            from .cloudflare import CloudflareJevTransport
            transport = CloudflareJevTransport(os.environ["CLOUDFLARE_ACCOUNT_ID"], os.environ["CLOUDFLARE_API_TOKEN"])
            _client = AsyncTypeSafeClient(api_key="cloudflare", model=C.TYPESAFE_MODEL, transport=transport)
        elif C.BACKEND == "typesafe":
            _client = AsyncTypeSafeClient(model=C.TYPESAFE_MODEL)
        else:
            raise ValueError(f"Unknown FASTGATE_BACKEND={C.BACKEND!r}; use typesafe, cloudflare or mock")
    return _client


@dataclass
class Decision:
    language: str
    intent: str
    intent_conf: float
    intent_probs: dict
    urgent: float
    wants_human: float
    passage_scores: list = field(default_factory=list)  # [{id, title, text, relevance, kept}]
    route: str = ""
    reason: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    model: str = ""

    @property
    def kept_passages(self) -> list[dict]:
        return [p for p in self.passage_scores if p["kept"]]

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * C.JEV_USD_PER_M_INPUT / 1e6


def triage_questions():
    return {
        "language": Choice(instructions="The language `query` is written in", criteria=C.LANGUAGES),
        "intent": Choice(instructions="Which university service team should handle `query`", criteria=C.INTENTS),
        "urgent": Noul(instructions="`query` says the matter is urgent or has a deadline within days"),
        "wants_human": Noul(instructions="`query` explicitly asks to speak to a person or staff member"),
    }


def grounding_questions(n: int):
    return {
        f"p{i}": Noul(instructions=f"`passages[{i}]` contains information that directly answers `query`")
        for i in range(n)
    }


def route(intent: str, conf: float, wants_human: float, n_kept: int) -> tuple[str, str]:
    """All routing is deterministic, auditable code. Order matters."""
    if wants_human >= C.HUMAN_ASK:
        return "human", "student asked for a person"
    if conf < C.AUTO_CONF:
        return "human_review", f"intent confidence {conf:.2f} < {C.AUTO_CONF}"
    if intent == "out_of_scope":
        return "template_decline", "confidently out of scope"
    if n_kept == 0:
        return "human_review", "no retrieved passage answers the query"
    return "llm_answer", f"grounded in {n_kept} passage(s)"


async def decide(query: str, passages: list[dict], client: AsyncTypeSafeClient | None = None) -> Decision:
    client = client or get_client()
    texts = [p["text"] for p in passages]
    t0 = time.perf_counter()
    calls = [client.system_one(state={"query": query}, questions=triage_questions())]
    if texts:
        calls.append(client.system_one(state={"query": query, "passages": texts},
                                       questions=grounding_questions(len(texts))))
    results = await asyncio.gather(*calls)
    latency = (time.perf_counter() - t0) * 1000

    tri = results[0].answers
    scores = []
    if texts:
        g = results[1].answers
        for i, p in enumerate(passages):
            rel = float(g[f"p{i}"].noul)
            scores.append({"id": p.get("id", str(i)), "title": p.get("title", ""), "text": p["text"],
                           "relevance": round(rel, 3), "kept": rel >= C.PASSAGE_KEEP})

    intent = tri["intent"]
    n_kept = sum(s["kept"] for s in scores)
    r, why = route(intent.choice, float(intent.confidence), float(tri["wants_human"].noul), n_kept)
    return Decision(
        language=tri["language"].choice,
        intent=intent.choice,
        intent_conf=round(float(intent.confidence), 3),
        intent_probs={k: round(float(v), 3) for k, v in intent.probabilities.items()},
        urgent=round(float(tri["urgent"].noul), 3),
        wants_human=round(float(tri["wants_human"].noul), 3),
        passage_scores=scores,
        route=r,
        reason=why,
        latency_ms=round(latency, 1),
        input_tokens=sum(x.usage.input_tokens for x in results if x.usage),
        model=results[0].model,
    )
