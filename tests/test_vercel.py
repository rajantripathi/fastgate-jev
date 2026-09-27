"""Vercel configuration and TypeSafe-compatible wire contract, offline only."""
import asyncio
import json

import httpx2
from typesafe_sdk import AsyncTypeSafeClient, Noul

from fastgate import config, core


def test_backend_precedence(monkeypatch):
    for name in ("FASTGATE_MOCK", "FASTGATE_BACKEND", "TYPESAFE_API_KEY",
                 "AI_GATEWAY_API_KEY", "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"):
        monkeypatch.delenv(name, raising=False)
    assert config._backend() == "mock"
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "test-only")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")
    assert config._backend() == "cloudflare"
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-only")
    assert config._backend() == "vercel"
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-only")
    assert config._backend() == "typesafe"
    monkeypatch.setenv("FASTGATE_BACKEND", "vercel")
    assert config._backend() == "vercel"
    monkeypatch.setenv("FASTGATE_MOCK", "1")
    assert config._backend() == "mock"


def test_vercel_client_wire_contract(monkeypatch):
    def handle(request):
        assert str(request.url) == "https://ai-gateway.vercel.sh/typesafe/v1/systemone"
        assert request.headers["authorization"] == "Bearer test-only"
        body = json.loads(request.content)
        assert body["model"] == "jev-latest"
        assert body["state"] == {"query": "A deadline tomorrow"}
        assert body["questions"]["urgent"]["type"] == "noul"
        return httpx2.Response(200, json={
            "model": "jev-1.13.0",
            "answers": {"urgent": {"type": "noul", "noul": 0.9}},
            "usage": {"input_tokens": 42, "output_tokens": 1},
        })

    def make_client(**kwargs):
        return AsyncTypeSafeClient(**kwargs, transport=httpx2.MockTransport(handle))

    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-only")
    monkeypatch.setattr(config, "BACKEND", "vercel")
    monkeypatch.setattr(config, "TYPESAFE_MODEL", "jev-latest")
    monkeypatch.setattr(core, "_client", None)
    monkeypatch.setattr(core, "AsyncTypeSafeClient", make_client)

    async def run():
        client = core.get_client()
        try:
            result = await client.system_one(
                state={"query": "A deadline tomorrow"},
                questions={"urgent": Noul(instructions="Is there a deadline?")},
            )
            assert result.model == "jev-1.13.0"
            assert result.answers["urgent"].noul == 0.9
            assert result.usage.input_tokens == 42
        finally:
            await client.aclose()

    asyncio.run(run())
