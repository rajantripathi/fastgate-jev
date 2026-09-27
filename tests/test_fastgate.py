"""Offline tests: run with  FASTGATE_MOCK=1 pytest -q"""
import asyncio
import os

os.environ["FASTGATE_MOCK"] = "1"

import numpy as np  # noqa: E402

from benchmark import coverage_at_precision, ece  # noqa: E402
from fastgate import KnowledgeBase, decide, route  # noqa: E402

KB = KnowledgeBase.load()


def test_route_order():
    assert route("fees", 0.99, 0.9, 2)[0] == "human"            # explicit ask wins
    assert route("fees", 0.40, 0.0, 2)[0] == "human_review"     # low confidence
    assert route("out_of_scope", 0.95, 0.0, 0)[0] == "template_decline"
    assert route("fees", 0.95, 0.0, 0)[0] == "human_review"     # ungrounded
    assert route("fees", 0.95, 0.0, 1)[0] == "llm_answer"


def test_retrieval_crosses_languages():
    for q, want in [("Kontrakt to'lovini bo'lib to'lash mumkinmi?", "fees-01"),
                    ("Как сделать регистрацию?", "visa-01"),
                    ("I forgot my password", "it-01")]:
        assert want in [p["id"] for p in KB.search(q, k=4)], q


def test_decide_end_to_end_mock():
    q = "Kontrakt to'lovini bo'lib to'lash mumkinmi? Muddat ertaga tugaydi."
    d = asyncio.run(decide(q, KB.search(q, k=4)))
    assert d.language == "uz" and d.intent == "fees"
    assert d.urgent > 0.5 and d.route == "llm_answer"
    assert any(p["id"] == "fees-01" for p in d.kept_passages)
    assert d.input_tokens > 0 and d.model.startswith("mock")


def test_out_of_scope_and_handoff():
    d1 = asyncio.run(decide("What's the best restaurant near the metro?", []))
    assert d1.route == "template_decline"
    q = "Срочно! Соедините меня с сотрудником, завтра дедлайн по оплате"
    d2 = asyncio.run(decide(q, KB.search(q)))
    assert d2.language == "ru" and d2.route == "human"


def test_metrics():
    conf, correct = np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 1, 1, 0])
    assert coverage_at_precision(conf, correct, 0.95) == 0.75
    assert abs(ece([1.0, 1.0], [1, 1])) < 1e-9
