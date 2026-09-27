"""Independent benchmark: Jev intent triage on EN / UZ / RU, optionally vs an LLM router.

    python benchmark.py                          # Jev only, data/queries.csv
    FASTGATE_LLM=anthropic:claude-haiku-4-5-20251001 python benchmark.py --llm-usd-per-query 0.0004

Writes results/results.csv, results/summary.md, results/reliability.png
"""
import argparse
import asyncio
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fastgate import config as C
from fastgate import llm
from fastgate.core import get_client, triage_questions

OUT = Path("results")
SEM = asyncio.Semaphore(8)  # be polite to early-access rate limits


async def run_jev(q: str):
    import time
    async with SEM:
        t0 = time.perf_counter()
        r = await get_client().system_one(state={"query": q}, questions=triage_questions())
        a = r.answers
        return (a["intent"].choice, float(a["intent"].confidence), max(a["intent"].probabilities.values()),
                a["language"].choice, (time.perf_counter() - t0) * 1000,
                r.usage.input_tokens if r.usage else 0, r.model)


async def run_llm(q: str):
    async with SEM:
        return await llm.llm_route(q)


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ((conf >= lo) if lo == 0 else (conf > lo)) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def coverage_at_precision(conf, correct, target=0.95):
    """Exploratory intent coverage at an observed precision, using whole ties.

    The threshold is selected on these same labels. This is not an estimate
    of held-out routing safety or the coverage of the full decision pipeline.
    """
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if conf.ndim != 1 or correct.ndim != 1 or len(conf) != len(correct):
        raise ValueError("conf and correct must be equally sized one-dimensional arrays")
    if not 0 < target <= 1:
        raise ValueError("target must be in (0, 1]")
    if not np.all(np.isfinite(conf)) or np.any((conf < 0) | (conf > 1)):
        raise ValueError("confidence must be finite and in [0, 1]")
    if not np.all(np.isin(correct, [0, 1])):
        raise ValueError("correct must contain only 0 or 1")
    if not len(conf):
        return 0.0
    order = np.argsort(-conf, kind="stable")
    scores, c = conf[order], correct[order]
    prec = np.cumsum(c) / np.arange(1, len(c) + 1)
    threshold_ends = np.r_[scores[:-1] != scores[1:], True]
    ok = np.flatnonzero(threshold_ends & (prec >= target))
    return float((ok.max() + 1) / len(c)) if len(ok) else 0.0


def reliability_plot(df, systems, path):
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    edges = np.linspace(0, 1, 11)
    colours = {"jev": "#003432", "llm": "#80003A"}
    for s in systems:
        conf, corr = df[f"{s}_conf"].to_numpy(), df[f"{s}_correct"].to_numpy(float)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = ((conf >= lo) if lo == 0 else (conf > lo)) & (conf <= hi)
            if m.sum() >= 3:
                xs.append(conf[m].mean()); ys.append(corr[m].mean())
        ax.plot(xs, ys, "o-", color=colours[s], label=s.upper())
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel="stated confidence", ylabel="observed accuracy",
           title="Calibration · EN / UZ / RU intent triage")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


async def main(csv, llm_usd_per_query, target):
    global OUT
    if C.MOCK:  # never let mock numbers land where real, committable results go
        OUT = OUT / "mock"
        print("MOCK backend: writing to results/mock/ (gitignored). These are NOT Jev numbers.")
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv)
    print(f"{len(df)} queries · backend: {C.BACKEND}{' (NOT Jev!)' if C.MOCK else ''}")

    jev = await asyncio.gather(*(run_jev(q) for q in df["query"]))
    df[["jev_pred", "jev_conf", "jev_maxprob", "jev_lang", "jev_ms", "jev_tokens", "jev_model"]] = pd.DataFrame(jev)
    model_ver = ", ".join(sorted(df["jev_model"].astype(str).unique()))
    df["jev_correct"] = df["jev_pred"] == df["gold_intent"]
    df["jev_lang_correct"] = df["jev_lang"] == df["language"]
    systems = ["jev"]

    if llm.available():
        res = await asyncio.gather(*(run_llm(q) for q in df["query"]), return_exceptions=True)
        res = [r if not isinstance(r, Exception) else ("error", 0.0, np.nan) for r in res]
        df[["llm_pred", "llm_conf", "llm_ms"]] = pd.DataFrame(res)
        df["llm_correct"] = df["llm_pred"] == df["gold_intent"]
        systems.append("llm")

    df.to_csv(OUT / "results.csv", index=False)

    rows = []
    for s in systems:
        row = {"system": s.upper() if s == "llm" else f"JEV ({model_ver})"}
        row["acc_all"] = df[f"{s}_correct"].mean()
        for lang in ("en", "uz", "ru"):
            row[f"acc_{lang}"] = df.loc[df.language == lang, f"{s}_correct"].mean()
        row["ECE"] = ece(df[f"{s}_conf"], df[f"{s}_correct"])
        row[f"intent_coverage@{target*100:g}%P"] = coverage_at_precision(df[f"{s}_conf"], df[f"{s}_correct"], target)
        accepted = df[f"{s}_conf"] >= C.AUTO_CONF
        row["fixed_conf"] = C.AUTO_CONF
        row["fixed_intent_coverage"] = accepted.mean()
        row["fixed_intent_precision"] = df.loc[accepted, f"{s}_correct"].mean()
        row["p50_ms"] = df[f"{s}_ms"].median()
        row["p95_ms"] = df[f"{s}_ms"].quantile(0.95)
        row["usd_per_1k"] = (df["jev_tokens"].mean() * C.JEV_USD_PER_M_INPUT / 1e6 * 1000 if s == "jev"
                             else (llm_usd_per_query * 1000 if llm_usd_per_query else np.nan))
        rows.append(row)
    summary = pd.DataFrame(rows).round(6)

    errors = df.loc[~df["jev_correct"], ["query", "language", "gold_intent", "jev_pred", "jev_conf"]]
    md = ["# FastGate benchmark", "",
          f"- Queries: {len(df)} · backend: {C.BACKEND} · model: {model_ver}"
          + (" · MOCK HEURISTIC, NOT JEV" if C.MOCK else ""),
          "- Scope: intent triage only. Latency and token cost exclude retrieval, grounding and answer generation.",
          "- Coverage at target precision is exploratory: the threshold is selected on this same sample, with tied scores kept together.",
          "- Fixed-threshold columns use FASTGATE_AUTO_CONF; they measure intent acceptance, not complete routing safety.",
          "- Cost is an estimate at the configured input-token price, not a provider invoice.",
          "- A small pilot does not establish 95% precision on future traffic. Validate on independently labelled held-out queries.",
          f"- Jev language-ID accuracy: {df['jev_lang_correct'].mean():.3f}", "",
          summary.to_markdown(index=False), "", "## Jev errors", "",
          errors.to_markdown(index=False) if len(errors) else "_none_"]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    reliability_plot(df, systems, OUT / "reliability.png")
    print(summary.to_string(index=False))
    print(f"\nWrote {OUT}/summary.md, results.csv, reliability.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="data/queries.csv")
    ap.add_argument("--llm-usd-per-query", type=float, default=0.0,
                    help="Measure from your provider bill for the cost column")
    ap.add_argument("--target-precision", type=float, default=0.95)
    a = ap.parse_args()
    asyncio.run(main(a.csv, a.llm_usd_per_query, a.target_precision))
