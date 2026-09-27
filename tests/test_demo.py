"""Evidence gates and truthful caption behaviour; no browser or live calls."""
import copy
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from record_demo import QUERIES, require_live, validate_decision
from render_demo import clip_plan, outcome_caption
from fastgate import config


def evidence():
    return {
        "run_id": "test-only", "query_sha256": hashlib.sha256(QUERIES[0].encode()).hexdigest(),
        "backend": "vercel", "model": "jev-test", "route": "llm_answer", "language": "uz",
        "intent": "fees", "intent_conf": 0.9, "latency_ms": 250.4, "display_latency_ms": 250,
        "urgent": 0.9, "wants_human": 0.0, "input_tokens": 100, "estimated_cost_usd": 0.0000042,
        "kept_ids": ["fees-01"], "answer_called": False, "answer_succeeded": False,
        "baseline_called": False,
    }


def test_recorder_refuses_mock_before_startup(monkeypatch):
    monkeypatch.setattr(config, "MOCK", True)
    with pytest.raises(ValueError, match="MOCK recording is disabled"):
        require_live()


def test_evidence_rejects_mock_stale_or_unexpected_fields():
    valid = evidence()
    validate_decision(valid, QUERIES[0], "vercel")
    for update in ({"model": "mock-jev"}, {"backend": "mock"},
                   {"query_sha256": "stale"}, {"unapproved_field": "not allowed"}):
        changed = copy.deepcopy(valid)
        changed.update(update)
        with pytest.raises(ValueError):
            validate_decision(changed, QUERIES[0], "vercel")


def test_handoff_caption_does_not_claim_real_delivery_or_hide_llm_calls():
    decision = evidence()
    decision["route"] = "human"
    assert outcome_caption(2, decision) == "Human handoff selected (demo)\nNo LLM call"
    decision["baseline_called"] = True
    assert "No LLM call" not in outcome_caption(2, decision)


def test_answer_caption_tracks_urgency_sources_and_disabled_generation():
    decision = evidence()
    decision["urgent"] = 0.2
    assert outcome_caption(0, decision) == "250 ms · 1 passage kept\nUrgency not flagged"
    assert "Answer generation is off" in outcome_caption(1, decision)
    decision["kept_ids"].append("fees-02")
    decision.update(answer_called=True, answer_succeeded=True)
    assert outcome_caption(1, decision) == "Jev kept 2 passages\nUsed as sources for the answer"


def test_edit_shortens_only_waiting_interval():
    record = {"start": 5, "click": 8, "result": 30, "end": 34}
    intervals, switch, shortened = clip_plan(record, 14)
    assert shortened
    assert intervals == [(5, 8.25), (29.85, 34)]
    assert switch >= 2.5
    assert intervals[-1][1] - record["result"] == 4


def test_edit_refuses_to_speed_up_visible_content():
    with pytest.raises(ValueError, match="exceeds the video slot"):
        clip_plan({"start": 0, "click": 20, "result": 25, "end": 30}, 8)
