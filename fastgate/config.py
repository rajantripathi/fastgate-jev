"""Central configuration. Thresholds live in code so they are auditable and tunable."""
import os

TYPESAFE_MODEL = os.getenv("TYPESAFE_MODEL", "jev-latest")

# Mock mode runs fully offline with a keyword heuristic that imitates Jev's
# response shape. It exists for tests and first-run UX only. Never record a demo in it.
MOCK = os.getenv("FASTGATE_MOCK") == "1" or not os.getenv("TYPESAFE_API_KEY")

# Optional LangChain model id for answer generation and the LLM-router baseline,
# e.g. "anthropic:claude-haiku-4-5-20251001" or "openai:gpt-4o-mini".
LLM = os.getenv("FASTGATE_LLM", "")

# Routing thresholds: tune these from benchmark.py output, not by intuition.
AUTO_CONF = float(os.getenv("FASTGATE_AUTO_CONF", "0.85"))
PASSAGE_KEEP = float(os.getenv("FASTGATE_PASSAGE_KEEP", "0.5"))
HUMAN_ASK = float(os.getenv("FASTGATE_HUMAN_ASK", "0.5"))
TOP_K = int(os.getenv("FASTGATE_TOP_K", "4"))

# Early-access list price (input tokens; output free). Re-check before publishing numbers.
JEV_USD_PER_M_INPUT = 0.042

LANGUAGES = {
    "en": "English",
    "uz": "Uzbek, written in Latin or Cyrillic script",
    "ru": "Russian",
    "mixed": "Two or more languages mixed in one message",
}

INTENTS = {
    "admissions": "Applying, entry requirements, application deadlines, admission documents",
    "fees": "Tuition amounts, payment plans, scholarships, refunds",
    "visa_registration": "Student visa, residence registration, invitation letters",
    "academic": "Courses, timetable, grades, exams, retakes, programme rules",
    "it_support": "Passwords, LMS access, university email, Wi-Fi",
    "out_of_scope": "Not about studying at or working with the university",
}
