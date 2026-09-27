"""Offline stand-in for the TypeSafe API, for tests and first-run UX.

It is a keyword heuristic that returns responses in Jev's exact wire format,
so the whole pipeline (SDK parsing, routing, UI) runs without a key.
It is NOT Jev. Do not report its numbers or record a demo with it.
"""
import json
import math
import re

import httpx2

INTENT_KEYWORDS = {
    "admissions": ["apply", "application", "admission", "entry requirement", "ielts", "enrol",
                   "qabul", "hujjat", "ariza", "поступ", "приём", "прием", "документ"],
    "fees": ["tuition", "fee", "pay", "instalment", "installment", "scholarship", "refund", "cost",
             "kontrakt", "to'lov", "stipendiya", "grant", "chegirma", "оплат", "контракт", "стипенд",
             "возврат", "стоимост", "скидк"],
    "visa_registration": ["visa", "registration", "residence", "invitation", "register my address",
                          "viza", "ro'yxat", "propiska", "виза", "визу", "регистрац", "пропис", "приглаш"],
    "academic": ["course", "timetable", "schedule", "grade", "exam", "credit", "retake", "module",
                 "dars", "jadval", "imtihon", "baho", "qayta topshir", "расписан", "экзамен", "оценк",
                 "пересдач", "предмет"],
    "it_support": ["password", "lms", "email", "wi-fi", "wifi", "login", "log in", "moodle",
                   "parol", "pochta", "пароль", "почт", "логин", "вай-фай", "wi fi"],
}
URGENT = ["urgent", "asap", "tomorrow", "today", "deadline", "immediately", "ertaga", "bugun",
          "tezda", "shoshilinch", "muddat", "срочно", "завтра", "сегодня", "дедлайн"]
HUMAN = ["human", "person", "staff", "operator", "someone real", "xodim", "odam", "operator",
         "человек", "сотрудник", "оператор", "живым"]
UZ_LATIN = ["salom", "qanday", "mumkin", "kerak", "qachon", "qayerda", "iltimos", "rahmat",
            "o'", "g'", "men ", "uchun", "bilan", "emas", "yo'q"]
UZ_CYR = set("ўқғҳ")


def _hits(text, words):
    t = text.lower().replace("ʻ", "'").replace("’", "'")
    return sum(w in t for w in words)


def _language(q):
    cyr = len(re.findall(r"[а-яёўқғҳ]", q.lower()))
    lat = len(re.findall(r"[a-z]", q.lower()))
    uz = _hits(q, UZ_LATIN) >= 1 or any(c in UZ_CYR for c in q.lower())
    if cyr and lat and cyr > 5 and lat > 5:
        return "mixed"
    if cyr > lat:
        return "uz" if any(c in UZ_CYR for c in q.lower()) else "ru"
    return "uz" if uz else "en"


def _intent_probs(text):
    raw = {k: _hits(text, v) for k, v in INTENT_KEYWORDS.items()}
    raw["out_of_scope"] = 1.5 if sum(raw.values()) == 0 else 0.0
    ex = {k: math.exp(2.5 * v) for k, v in raw.items()}
    z = sum(ex.values())
    return {k: v / z for k, v in ex.items()}


def _answer(state, q):
    query = state.get("query", "") if isinstance(state, dict) else str(state)
    crit, instr = q.get("criteria"), q.get("instructions", "")
    if q["type"] == "choice":
        if isinstance(crit, dict) and "uz" in crit:
            choice = _language(query)
            probs = {k: (0.9 if k == choice else 0.1 / (len(crit) - 1)) for k in crit}
        else:
            probs = _intent_probs(query)
            probs = {k: probs.get(k, 0.0) for k in crit}
            choice = max(probs, key=probs.get)
        return {"type": "choice", "choice": choice, "confidence": round(max(probs.values()), 3),
                "probabilities": {k: round(v, 3) for k, v in probs.items()}}
    # noul
    m = re.search(r"passages\[(\d+)\]", instr)
    if m:
        passage = state["passages"][int(m.group(1))]
        qi, pi = _intent_probs(query), _intent_probs(passage)
        same = max(qi, key=qi.get) == max(pi, key=pi.get) and max(qi, key=qi.get) != "out_of_scope"
        return {"type": "noul", "noul": 0.9 if same else 0.08}
    if "urgent" in instr:
        return {"type": "noul", "noul": 0.95 if _hits(query, URGENT) else 0.05}
    if "person" in instr:
        return {"type": "noul", "noul": 0.95 if _hits(query, HUMAN) else 0.03}
    return {"type": "noul", "noul": 0.5}


def _handler(request: httpx2.Request) -> httpx2.Response:
    body = json.loads(request.content)
    answers = {k: _answer(body["state"], q) for k, q in body["questions"].items()}
    tokens = len(json.dumps(body)) // 4
    return httpx2.Response(200, json={"model": "mock-jev (offline heuristic)", "answers": answers,
                                      "usage": {"input_tokens": tokens, "output_tokens": 0}})


def mock_transport() -> httpx2.MockTransport:
    return httpx2.MockTransport(_handler)
