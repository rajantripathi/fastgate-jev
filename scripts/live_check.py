"""10-second live check: one real Jev call through whichever backend is configured.

    python scripts/live_check.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastgate import KnowledgeBase, decide  # noqa: E402
from fastgate import config as C  # noqa: E402

Q = "Kontrakt to'lovini bo'lib to'lash mumkinmi? Muddat ertaga tugaydi."


async def main():
    print(f"backend: {C.BACKEND}")
    if C.MOCK:
        print("No credentials found, so this would only test the offline mock.\n"
              "Set CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN (or TYPESAFE_API_KEY) and re-run.")
        sys.exit(1)
    d = await decide(Q, KnowledgeBase.load().search(Q))
    print(f"model:    {d.model}")
    print(f"language: {d.language} · intent: {d.intent} ({d.intent_conf}) · urgent: {d.urgent}")
    print(f"route:    {d.route} · {d.reason}")
    print(f"latency:  {d.latency_ms:.0f} ms · tokens: {d.input_tokens}")
    print("LIVE OK" if d.model.startswith("jev") else "WARNING: unexpected model string")


asyncio.run(main())
