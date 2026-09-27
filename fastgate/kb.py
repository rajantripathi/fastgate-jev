"""Tiny multilingual-friendly BM25 retriever.

Stems are 5-character prefixes, a cheap trick that survives Uzbek and Russian
suffixes ("to'lovini" and "to'lov" both become "to'lo"). Each passage also
carries UZ/RU keywords. For production, swap in a multilingual embedding
retriever (e.g. your existing vector store); FastGate only needs `search()`.
"""
import json
import math
import re
from collections import Counter
from pathlib import Path

_TOKEN = re.compile(r"[\w'ʻ’]+", re.UNICODE)


def stems(text: str) -> list[str]:
    toks = [t.replace("ʻ", "'").replace("’", "'") for t in _TOKEN.findall(text.lower())]
    return [t[:5] for t in toks if len(t) > 2]


class KnowledgeBase:
    def __init__(self, passages: list[dict], k1: float = 1.4, b: float = 0.75):
        self.passages = passages
        self.docs = [stems(p["text"] + " " + " ".join(p.get("keywords", []))) for p in passages]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))
        df = Counter(s for d in self.docs for s in set(d))
        n = len(self.docs)
        self.idf = {s: math.log(1 + (n - c + 0.5) / (c + 0.5)) for s, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    @classmethod
    def load(cls, path: str | Path = Path(__file__).parent.parent / "data" / "kb.json"):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def search(self, query: str, k: int = 4) -> list[dict]:
        q = stems(query)
        scored = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = sum(
                self.idf.get(t, 0) * tf[t] * (self.k1 + 1)
                / (tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                for t in q if t in tf
            )
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [{**self.passages[i], "bm25": round(s, 3)} for s, i in scored[:k]]
