"""FastGate demo UI.  Run:  python app.py   (set TYPESAFE_API_KEY for real Jev)."""
import asyncio

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
    "human_review": "I have passed your message to a member of staff, who will reply shortly.",
    "human": "Connecting you with a member of staff now.",
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
                "<b>MOCK MODE</b>: offline keyword heuristic, not Jev. Set TYPESAFE_API_KEY for real results.</div>")
    extra = f" · LLM: {C.LLM}" if llm.available() else " · LLM step off (set FASTGATE_LLM)"
    return (f"<div style='padding:8px 12px;border-radius:6px;background:#AADED9;color:#21201E'>"
            f"<b>LIVE</b> · backend: {C.BACKEND} · model: {C.TYPESAFE_MODEL}{extra}</div>")


async def run(query: str):
    if not query.strip():
        raise gr.Error("Type a student message first.")
    passages = KB.search(query, k=C.TOP_K)

    tasks = [decide(query, passages)]
    if llm.available():
        tasks.append(llm.llm_route(query))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    d = results[0]
    if isinstance(d, Exception):
        raise gr.Error(f"TypeSafe call failed: {d}")

    colour, label = ROUTE_STYLE[d.route]
    race = ""
    if len(results) > 1 and not isinstance(results[1], Exception):
        li, lc, lms = results[1]
        race = (f"<br><span style='opacity:.85'>LLM router for comparison: {li} "
                f"(self-reported {lc:.2f}) in {lms:.0f} ms</span>")
    badge = (f"<div style='padding:14px;border-radius:8px;background:{colour};color:#F0EFE3;font-size:15px'>"
             f"<b>{label}</b> · {d.reason}<br>"
             f"Jev decided in <b>{d.latency_ms:.0f} ms</b> · language <b>{d.language}</b> · "
             f"urgent {d.urgent:.2f} · wants human {d.wants_human:.2f} · "
             f"cost ${d.cost_usd:.7f} · {d.model}{race}</div>")

    rows = [[p["id"], p["title"], p["relevance"], "yes" if p["kept"] else "no"] for p in d.passage_scores]

    if d.route == "llm_answer":
        if llm.available():
            try:
                reply = await llm.answer(query, d.language, d.kept_passages)
            except Exception as e:  # keep the demo alive
                reply = f"_LLM call failed: {e}_"
        else:
            reply = ("_LLM step off. It would answer from:_\n\n"
                     + "\n".join(f"- **{p['id']}**: {p['text']}" for p in d.kept_passages))
    else:
        reply = TEMPLATES[d.route]

    raw = {k: v for k, v in d.__dict__.items() if k != "passage_scores"}
    return badge, d.intent_probs, rows, reply, raw


with gr.Blocks(title="FastGate · Jev decision layer") as demo:
    gr.Markdown("# FastGate\n**Jev decides. Code routes. The LLM only writes when it should.**  "
                "A System One decision layer for a multilingual (EN / UZ / RU) university helpdesk.")
    gr.HTML(banner())
    with gr.Row():
        q = gr.Textbox(label="Student message", lines=2, scale=4, value=EXAMPLES[0])
        btn = gr.Button("Decide", variant="primary", scale=1)
    gr.Examples(EXAMPLES, inputs=q)
    badge = gr.HTML()
    with gr.Row():
        probs = gr.Label(label="Intent probabilities (Jev Choice)", num_top_classes=6)
        table = gr.Dataframe(headers=["passage", "title", "relevance (Noul)", "kept"],
                             label="Retrieved passages, gated by Jev")
    reply = gr.Markdown(label="Response")
    with gr.Accordion("Raw decision", open=False):
        raw = gr.JSON()
    btn.click(run, q, [badge, probs, table, reply, raw])
    q.submit(run, q, [badge, probs, table, reply, raw])

if __name__ == "__main__":
    demo.launch()
