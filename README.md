# FastGate

**Jev decides. Code routes. The LLM only writes when it should.**

![tests](https://github.com/rajantripathi/fastgate-jev/actions/workflows/ci.yml/badge.svg)

FastGate uses [Jev](https://typesafe.ai), TypeSafe AI's System One model, as the decision layer of a multilingual (English / Uzbek / Russian) university helpdesk. Jev returns typed, calibrated decisions instead of text. FastGate turns those decisions into deterministic, auditable routing, and calls a generative LLM only for queries that are confidently understood and grounded in retrieved sources.

It also ships an **independent benchmark** of Jev on low-resource-language triage: accuracy per language, calibration (ECE and a reliability diagram), exploratory intent coverage, latency and cost.

## How it works

```
student message
      │
      ├──► retriever (BM25 with Uzbek/Russian-tolerant stems) ──► top-k passages
      │
      ├──► Jev call 1 · triage     state = {query}
      │        language (Choice) · intent (Choice + confidence) · urgent (Noul) · wants_human (Noul)
      │
      └──► Jev call 2 · grounding  state = {query, passages}      (runs in parallel)
               one Noul per passage: "does passages[i] directly answer the query?"
                                   │
                    deterministic routing in code
     ┌──────────────┬──────────────┼──────────────────┬───────────────────┐
  wants human    confidence      out of scope      no grounded       confident +
  → handoff      < threshold     → template        passage           grounded
                 → human review                    → human review    → LLM answers from
                                                                       kept passages only
```

Design choices, each following TypeSafe's published guidance:

- **Atomic questions, composed in code.** No single "handle this ticket" prompt. Four narrow judgments, combined by `route()`.
- **Two calls, not one.** Triage never sees the passages, because unrelated context in the state lowers accuracy.
- **Confidence as a second axis.** The intent answer says *what*; its confidence decides *whether to act*.
- **No arithmetic or dates in the model.** Fee maths and deadlines stay in code.

## Quick start

```bash
git clone https://github.com/rajantripathi/fastgate-jev && cd fastgate-jev
pip install -r requirements.txt

# Option A: official API (https://console.typesafe.ai)
export TYPESAFE_API_KEY=...
# Option B: Jev on Cloudflare Workers AI (model typesafe/jev)
export CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_API_TOKEN=...
# Option C: Vercel AI Gateway (TypeSafe-compatible API)
export AI_GATEWAY_API_KEY=...

python scripts/live_check.py       # one live call, prints the Jev version
python app.py                      # demo UI at http://127.0.0.1:7860
python benchmark.py                # writes results/summary.md + reliability.png
```

Choose one option. Auto-detection checks TypeSafe, then Vercel, then Cloudflare. `FASTGATE_BACKEND=vercel` explicitly selects Vercel; `FASTGATE_MOCK=1` always selects the offline mock. `.env.example` lists settings, but `.env` is not loaded automatically.

The Vercel backend uses the official SDK with `base_url="https://ai-gateway.vercel.sh/typesafe"`, following [Vercel's TypeSafe client integration](https://vercel.com/changelog/ai-gateway-now-supports-typesafe-clients-and-http-api-for-jev). If a live check reports a model error, retry with `TYPESAFE_MODEL=typesafe-ai/jev`. A 401 indicates an authentication problem; a 402 indicates a billing or credit problem.

The Cloudflare backend is a thin transport adapter under the official `typesafe-sdk`: same request types, same response parsing, only the network hop changes (`fastgate/cloudflare.py`).

With no key, FastGate starts in **mock mode**: an offline keyword heuristic that speaks Jev's wire format so the pipeline, UI and tests run anywhere. The UI shows a red banner in this mode. Mock numbers are not Jev numbers.

Optional LLM for answers and the LLM-router baseline (any LangChain model id):

```bash
pip install langchain langchain-anthropic
export FASTGATE_LLM=anthropic:claude-haiku-4-5-20251001 ANTHROPIC_API_KEY=...
python benchmark.py --llm-usd-per-query 0.0004   # measure this from your bill
```

## Results

> **Pilot: 36 queries.** Run `python benchmark.py` with a real key and paste `results/summary.md` here. State the exact model version.

| system | acc EN | acc UZ | acc RU | ECE ↓ | intent coverage @95% observed precision | p50 ms | $ / 1k |
|---|---|---|---|---|---|---|---|
| Jev (`jev-1.x`) | | | | | | | |
| LLM router | | | | | | | |

The benchmark measures the intent-triage call only. Its latency and estimated token cost exclude retrieval, the grounding call and answer generation. Coverage at 95% observed precision selects a confidence threshold on the same pilot sample, keeping tied scores together. It does not establish safe automatic routing or 95% precision on future traffic. Fixed-threshold intent coverage and precision are also reported at `FASTGATE_AUTO_CONF` (default 0.85).

<!-- Restore after running the benchmark with a real key: ![calibration](results/reliability.png) -->

## Repository layout

```
fastgate/
  core.py     Jev questions, parallel calls, deterministic route()
  kb.py       small BM25 retriever with suffix-tolerant stems
  llm.py      optional LangChain answer generator + LLM-router baseline
  mock.py     offline stand-in speaking Jev's wire format (tests only)
  cloudflare.py  SDK transport adapter for Jev on Cloudflare Workers AI
  config.py   backend selection, thresholds, intents, languages
app.py        Gradio demo
benchmark.py  accuracy per language, ECE, coverage@precision, latency, cost
data/         sample knowledge base (fictional university) + 36 labelled EN/UZ/RU queries
tests/        offline tests (CI runs them in mock mode)
```

## Limitations

- The knowledge base describes a fictional university. Policies are illustrative only.
- 36 sample queries show the pipeline; publishable claims need a larger, independently labelled set (300+ is a sensible minimum).
- Jev is in early access. TypeSafe has not published architecture or weights, and pricing and behaviour may change between versions.
- Adversarial content in the state can influence Jev's answers. Test prompt-injection cases before any real deployment.

## Deploy as a Hugging Face Space

Create a Gradio Space, push this repo, and add `AI_GATEWAY_API_KEY`, `TYPESAFE_API_KEY`, or the two `CLOUDFLARE_*` secrets (and optionally `FASTGATE_LLM` plus the provider key) under *Settings → Secrets*.

## Author

**Dr Rajan Prasad Tripathi**, Director, AI² Innovation Lab, American University of Technology (Tashkent) · NVIDIA DLI Certified Instructor
[GitHub](https://github.com/rajantripathi) · ORCID [0000-0002-1192-4773](https://orcid.org/0000-0002-1192-4773)

MIT licence.
