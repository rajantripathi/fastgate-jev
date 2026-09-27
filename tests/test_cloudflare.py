"""Cloudflare adapter tests against a fake Cloudflare endpoint (offline)."""
import asyncio
import json

import httpx2
import pytest
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, TypeSafeError

from fastgate.cloudflare import CloudflareJevTransport

# Response body from Cloudflare's typesafe/jev docs, wrapped in the REST envelope
DOC_RESULT = {
    "model": "jev-1.13.0",
    "answers": {
        "is_urgent": {"type": "noul", "noul": 0.95},
        "department": {"type": "choice", "choice": "billing", "confidence": 0.8,
                       "probabilities": {"billing": 0.87, "sales": 0, "technical": 0.13}},
    },
    "usage": {"input_tokens": 426, "output_tokens": 73},
}
QUESTIONS = {
    "is_urgent": Noul(instructions="Does this convey urgency?"),
    "department": Choice(instructions="Which team should handle this?",
                         criteria={"billing": "Payments", "technical": "Bugs", "sales": "Pricing"}),
}


def client_with(handler):
    t = CloudflareJevTransport("acct123", "tok456", inner=httpx2.MockTransport(handler))
    return AsyncTypeSafeClient(api_key="cloudflare", transport=t)


@pytest.mark.parametrize("wrapped", [True, False])
def test_request_rewrite_and_unwrap(wrapped):
    seen = {}

    def handler(req):
        seen["url"], seen["auth"] = str(req.url), req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        body = {"result": DOC_RESULT, "success": True, "errors": [], "messages": []} if wrapped else DOC_RESULT
        return httpx2.Response(200, json=body)

    r = asyncio.run(client_with(handler).system_one(
        state="Help! My payouts have been failing for 3 days.", questions=QUESTIONS))

    assert seen["url"] == "https://api.cloudflare.com/client/v4/accounts/acct123/ai/run"
    assert seen["auth"] == "Bearer tok456"
    assert seen["body"]["model"] == "typesafe/jev"
    assert set(seen["body"]["input"]) == {"state", "questions"}
    assert seen["body"]["input"]["questions"]["department"]["type"] == "choice"
    assert r.model == "jev-1.13.0"
    assert r.answers["department"].choice == "billing"
    assert r.answers["is_urgent"].noul == 0.95
    assert r.usage.input_tokens == 426


def test_cloudflare_error_surfaces_clearly():
    def handler(req):
        return httpx2.Response(401, json={"success": False, "result": None,
                                          "errors": [{"code": 10000, "message": "Authentication error"}]})

    with pytest.raises(TypeSafeError) as e:
        asyncio.run(client_with(handler).system_one(state="x", questions=QUESTIONS))
    assert "Authentication error" in str(e.value) or "401" in str(e.value)


def test_missing_credentials():
    with pytest.raises(ValueError):
        CloudflareJevTransport("", "")
