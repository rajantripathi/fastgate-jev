"""Record only the live local demo. Run in the terminal holding your API key.

    python scripts/record_demo.py

No terminal, provider dashboard, browser profile or environment is recorded.
The manifest contains only synthetic queries and an allowlist of UI decisions.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUERIES = [
    "Kontrakt to'lovini bo'lib to'lash mumkinmi? Muddat ertaga tugaydi.",
    "Как сделать регистрацию для иностранного студента?",
    "Срочно! Соедините меня с сотрудником, завтра дедлайн по оплате",
    "What's the best restaurant near the metro?",
]
FIELDS = {
    "run_id", "query_sha256", "backend", "model", "route", "language", "intent",
    "intent_conf", "latency_ms", "display_latency_ms", "urgent", "wants_human",
    "input_tokens", "estimated_cost_usd", "kept_ids", "answer_called",
    "answer_succeeded", "baseline_called",
}


def require_live():
    from fastgate import config as config
    names = {
        "typesafe": ("TYPESAFE_API_KEY",), "vercel": ("AI_GATEWAY_API_KEY",),
        "cloudflare": ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"),
    }
    if config.MOCK or config.BACKEND not in names:
        raise ValueError("Recording requires a live backend. MOCK recording is disabled.")
    if any(not os.getenv(name) for name in names[config.BACKEND]):
        raise ValueError("Set the selected backend's credentials privately in this terminal.")
    return config.BACKEND


def validate_decision(data, query, backend):
    if set(data) != FIELDS:
        raise ValueError("Unexpected UI evidence fields; update the recorder before publishing.")
    if data["backend"] != backend or not data["model"].startswith("jev-"):
        raise ValueError("Recording did not receive a verified live Jev model.")
    if data["query_sha256"] != hashlib.sha256(query.encode()).hexdigest():
        raise ValueError("Stale result: the badge belongs to a different query.")
    if data["latency_ms"] <= 0 or data["route"] not in {
        "human", "human_review", "template_decline", "llm_answer"
    }:
        raise ValueError("Invalid decision evidence.")


def wait_for_app(proc):
    until = time.monotonic() + 90
    while time.monotonic() < until:
        if proc.poll() is not None:
            raise RuntimeError("The local app exited. Run python app.py privately to diagnose it.")
        try:
            with urlopen("http://127.0.0.1:7860", timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.25)
    raise RuntimeError("The local app did not start within 90 seconds.")


async def capture(destination, backend, scale):
    from playwright.async_api import async_playwright
    raw = destination / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    records = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1080, "height": 1080}, device_scale_factor=scale,
            record_video_dir=str(raw), record_video_size={"width": 1080, "height": 1080},
            color_scheme="light", reduced_motion="reduce",
        )
        try:
            epoch = time.monotonic()
            page = await context.new_page()
            video = page.video
            await page.goto("http://127.0.0.1:7860", wait_until="networkidle")
            banner = await page.locator("#backend-banner").inner_text()
            if "LIVE" not in banner or "MOCK" in banner:
                raise RuntimeError("Refusing to record a page without the LIVE banner.")
            await page.evaluate("document.fonts.ready")
            box = page.locator("#student-query textarea")
            for index, query in enumerate(QUERIES):
                await page.evaluate("window.scrollTo(0, 0)")
                await box.fill("")
                await page.wait_for_timeout(600)
                record = {"query": query, "start": time.monotonic() - epoch}
                await box.press_sequentially(query, delay=35)
                record["click"] = time.monotonic() - epoch
                await page.locator("#decide-button").click()
                wanted = hashlib.sha256(query.encode()).hexdigest()
                await page.wait_for_function("""wanted => {
                    const node = document.querySelector('[data-fastgate-decision]');
                    if (!node) return false;
                    return JSON.parse(node.dataset.fastgateDecision).query_sha256 === wanted;
                }""", arg=wanted, timeout=120000)
                node = page.locator("[data-fastgate-decision]")
                data = json.loads(await node.get_attribute("data-fastgate-decision"))
                validate_decision(data, query, backend)
                await node.scroll_into_view_if_needed()
                record["result"] = time.monotonic() - epoch
                record["decision"] = data
                record["badge_text"] = await node.inner_text()
                await page.wait_for_timeout(3500)
                if index == 1:
                    await page.locator("#passage-table").scroll_into_view_if_needed()
                    record["table"] = time.monotonic() - epoch
                    await page.wait_for_timeout(3500)
                record["end"] = time.monotonic() - epoch
                await page.screenshot(path=str(raw / f"query-{index + 1}.png"))
                records.append(record)
            await page.close()
            video_path = await video.path()
        finally:
            await context.close()
            await browser.close()
    return video_path, records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "demo")
    ap.add_argument("--device-scale-factor", type=float, default=2)
    args = ap.parse_args()
    backend = require_live()  # Fail before opening a browser or writing any video.
    destination = args.output_dir.resolve()
    if (destination / "recording.json").exists():
        raise ValueError("recording.json already exists. Archive that run or choose --output-dir.")
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", 7860))
        except OSError:
            raise RuntimeError("Port 7860 is busy. Stop the existing app before recording.") from None
    child_env = os.environ.copy()
    child_env.update(GRADIO_SERVER_NAME="127.0.0.1", GRADIO_SERVER_PORT="7860",
                     GRADIO_ANALYTICS_ENABLED="False", FASTGATE_COMPARE_LLM="0")
    # Discard child logs so an SDK error cannot persist request headers or credentials.
    proc = subprocess.Popen([sys.executable, str(ROOT / "app.py")], cwd=ROOT,
                            env=child_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    try:
        wait_for_app(proc)
        path, records = asyncio.run(capture(destination, backend, args.device_scale_factor))
        manifest = {
            "schema": 1, "complete": True, "backend": backend,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "video": str(Path(path).resolve().relative_to(destination)), "queries": records,
            "note": "UI recording only. Staff handoffs are simulated. LLM-router comparison disabled.",
        }
        (destination / "recording.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print("Live recording complete. Evidence: " + str(destination / "recording.json"))
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        # Do not persist provider/browser exceptions that might contain sensitive context.
        print(f"Recording failed ({type(error).__name__}). No completed manifest was published.", file=sys.stderr)
        sys.exit(1)
