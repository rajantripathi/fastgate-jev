"""FastGate demo UI.  Run:  python app.py   (configure a live backend for real Jev)."""
import asyncio
import hashlib
import html
import json
import os
import uuid

import gradio as gr

from fastgate import KnowledgeBase, decide
from fastgate import config as C
from fastgate import llm

KB = KnowledgeBase.load()

ROUTE_STYLE = {
    "llm_answer": ("#003432", "LLM ANSWER"),
    "template_decline": ("#21201E", "TEMPLATE DECLINE"),
    "human_review": ("#80003A", "HUMAN REVIEW"),
    "human": ("#80003A", "HANDOFF TO STAFF"),
}
TEMPLATES = {
    "template_decline": "I can only help with questions about studying at the university.",
    "human_review": "Demo: this message would be queued for staff review. No message has been sent.",
    "human": "Demo: this request would be handed to staff. No staff connection has been made.",
}
EXAMPLES = [
    "Kontrakt to'lovini bo'lib to'lash mumkinmi? Muddat ertaga tugaydi.",
    "Как сделать регистрацию для иностранного студента?",
    "I forgot my LMS password and can't log in",
    "Imtihondan yiqildim, qayta topshirsam bo'ladimi?",
    "Срочно! Соедините меня с сотрудником, завтра дедлайн по оплате",
    "What's the best restaurant near the metro?",
]


def banner():
    if C.MOCK:
        return ("<div style='padding:8px 12px;border-radius:6px;background:#80003A;color:#F0EFE3'>"
                "<b style='color:inherit'>MOCK MODE</b>: offline keyword heuristic, not Jev. Set AI_GATEWAY_API_KEY or TYPESAFE_API_KEY for real results.</div>")
    extra = f" · LLM: {C.LLM}" if llm.available() else " · answer generation off"
    return (f"<div style='padding:8px 12px;border-radius:6px;background:#AADED9;color:#21201E'>"
            f"<b style='color:inherit'>LIVE</b> · backend: {C.BACKEND} · model: {C.TYPESAFE_MODEL}{extra}</div>")


async def run(query: str):
    if not query.strip():
        raise gr.Error("Type a student message first.")
    passages = KB.search(query, k=C.TOP_K)

    tasks = [decide(query, passages)]
    if llm.available() and os.getenv("FASTGATE_COMPARE_LLM", "1") == "1":
        tasks.append(llm.llm_route(query))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    d = results[0]
    if isinstance(d, Exception):
        raise gr.Error(f"{C.BACKEND} call failed ({type(d).__name__}). Check provider access privately.")

    colour, label = ROUTE_STYLE[d.route]
    if d.route == "llm_answer" and not llm.available():
        label = "ANSWER ELIGIBLE"
    race = ""
    if len(results) > 1 and not isinstance(results[1], Exception):
        li, lc, lms = results[1]
        race = (f"<br><span style='opacity:.85'>LLM router for comparison: {li} "
                f"(self-reported {lc:.2f}) in {lms:.0f} ms</span>")
    badge = (f"<div style='padding:14px;border-radius:8px;background:{colour};color:#F0EFE3;font-size:15px'>"
             f"<b style='color:inherit'>{label}</b> · {d.reason}<br>"
             f"Jev decided in <b style='color:inherit'>{d.latency_ms:.0f} ms</b> · language <b style='color:inherit'>{d.language}</b> · "
             f"urgent {d.urgent:.2f} · wants human {d.wants_human:.2f} · "
             f"cost ${d.cost_usd:.7f} · {d.model}{race}</div>")

    rows = [[p["id"], p["title"], p["relevance"], "yes" if p["kept"] else "no"] for p in d.passage_scores]

    answer_called = False
    answer_succeeded = False
    if d.route == "llm_answer":
        if llm.available():
            try:
                answer_called = True
                reply = await llm.answer(query, d.language, d.kept_passages)
                answer_succeeded = True
            except Exception as e:  # keep the demo alive
                reply = f"_Answer generation failed ({type(e).__name__})._"
        else:
            reply = ("_LLM step off. It would answer from:_\n\n"
                     + "\n".join(f"- **{p['id']}**: {p['text']}" for p in d.kept_passages))
    else:
        reply = TEMPLATES[d.route]

    evidence = {
        "run_id": uuid.uuid4().hex,
        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
        "backend": C.BACKEND, "model": d.model, "route": d.route,
        "language": d.language, "intent": d.intent, "intent_conf": d.intent_conf,
        "latency_ms": d.latency_ms, "display_latency_ms": round(d.latency_ms),
        "urgent": d.urgent, "wants_human": d.wants_human,
        "input_tokens": d.input_tokens, "estimated_cost_usd": d.cost_usd,
        "kept_ids": [p["id"] for p in d.kept_passages],
        "answer_called": answer_called, "answer_succeeded": answer_succeeded,
        "baseline_called": len(tasks) > 1,
    }
    badge = '<div data-fastgate-decision="' + html.escape(json.dumps(evidence), quote=True) + '">' + badge + '</div>'
    raw = {k: v for k, v in d.__dict__.items() if k != "passage_scores"}
    return badge, d.intent_probs, rows, reply, raw


with gr.Blocks(title="FastGate · Jev decision layer") as demo:
    gr.Markdown("# FastGate\n**Jev decides. Code routes. The LLM only writes when it should.**  "
                "A System One decision layer for a multilingual (EN / UZ / RU) university helpdesk.")
    gr.HTML(banner(), elem_id="backend-banner")
    with gr.Row():
        q = gr.Textbox(label="Student message", lines=2, scale=4, value=EXAMPLES[0], elem_id="student-query")
        btn = gr.Button("Decide", variant="primary", scale=1, elem_id="decide-button")
    gr.Examples(EXAMPLES, inputs=q)
    badge = gr.HTML(elem_id="decision-badge")
    with gr.Row():
        probs = gr.Label(label="Intent probabilities (Jev Choice)", num_top_classes=6, elem_id="intent-probabilities")
        table = gr.Dataframe(headers=["passage", "title", "Noul", "kept"],
                             label="Retrieved passages, gated by Jev", elem_id="passage-table",
                             column_widths=[105, 180, 70, 65])
    reply = gr.Markdown(label="Response")
    with gr.Accordion("Raw decision", open=False):
        raw = gr.JSON()
    btn.click(run, q, [badge, probs, table, reply, raw])
    q.submit(run, q, [badge, probs, table, reply, raw])

if __name__ == "__main__":
    demo.launch()
