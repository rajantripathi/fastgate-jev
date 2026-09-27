"""Run the official TypeSafe SDK against Jev on Cloudflare Workers AI.

The SDK and Cloudflare share Jev's wire format ({state, questions} in,
{model, answers, usage} out). This transport only rewrites the network hop:
  SDK POST https://api.typesafe.ai/v1/systemone {state, model, questions}
  ->  POST https://api.cloudflare.com/client/v4/accounts/<id>/ai/run
      {"model": "typesafe/jev", "input": {state, questions}}
and unwraps Cloudflare's {"result": ..., "success": ...} envelope, so SDK
parsing, typing and error classes stay exactly the same.

Env: CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN (Workers AI read + edit).
"""
import json
import os

import httpx2

CF_MODEL = os.getenv("CLOUDFLARE_JEV_MODEL", "typesafe/jev")


class CloudflareJevTransport(httpx2.AsyncBaseTransport):
    def __init__(self, account_id: str, api_token: str, inner: httpx2.AsyncBaseTransport | None = None):
        if not account_id or not api_token:
            raise ValueError("Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN")
        self.url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run"
        self.token = api_token
        self.inner = inner or httpx2.AsyncHTTPTransport(retries=2)

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(await request.aread() or b"{}")
        payload = {"model": CF_MODEL, "input": {"state": body["state"], "questions": body["questions"]}}
        cf_req = httpx2.Request(
            "POST", self.url, json=payload,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        resp = await self.inner.handle_async_request(cf_req)
        raw = await resp.aread()
        try:
            data = json.loads(raw)
        except ValueError:
            return httpx2.Response(resp.status_code, content=raw, request=request)

        ok = resp.status_code < 400 and data.get("success", True) is not False
        if not ok:
            errs = data.get("errors") or [{"message": raw.decode(errors="replace")[:300]}]
            msg = "; ".join(str(e.get("message", e)) for e in errs)
            status = resp.status_code if resp.status_code >= 400 else 502
            return httpx2.Response(status, json={"error": {"type": "cloudflare", "message": f"Cloudflare: {msg}"},
                                                 "message": f"Cloudflare: {msg}"}, request=request)

        result = data.get("result", data)  # REST wraps in {"result": ...}; be tolerant of both
        return httpx2.Response(200, json=result, request=request)

    async def aclose(self) -> None:
        await self.inner.aclose()
