"""Compose the silent demo from live UI evidence and the real benchmark.

Requires ffmpeg, ffprobe and Pillow. No credentials are read by this script.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

from PIL import Image, ImageDraw, ImageFont

from record_demo import QUERIES, validate_decision

ROOT = Path(__file__).resolve().parents[1]
GREEN = "#003432"
WHITE = "#F0EFE3"
MINT = "#AADED9"
SIZE = 1080


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    raise ValueError("Install Arial, Helvetica Neue or DejaVu Sans before rendering.")


def wrap(text, face, width):
    lines = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split():
            candidate = (current + " " + word).strip()
            if current and face.getlength(candidate) > width:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
    return lines


def card(path, title, body, footer="", caption=False):
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0) if caption else GREEN)
    draw = ImageDraw.Draw(canvas)
    face = font(48 if caption else 46)
    lines = wrap(body, face, 940)
    if caption and len(lines) > 2:
        raise ValueError("Caption exceeds two lines: " + body)
    if caption:
        height = 60 * len(lines) + 42
        top = 1040 - height
        draw.rounded_rectangle((25, top, 1055, 1055), radius=16, fill=(0, 52, 50, 225))
        y = top + 20
    else:
        draw.rectangle((70, 90, 200, 99), fill=MINT)
        draw.text((70, 125), title, font=font(32), fill=MINT)
        if len(lines) > 10:
            raise ValueError("Card text is too long to read in its allotted time.")
        y = max(240, (1080 - len(lines) * 64) / 2)
    for line in lines:
        draw.text(((1080 - face.getlength(line)) / 2, y), line, font=face, fill=WHITE)
        y += 60 if caption else 64
    if footer:
        small = font(25)
        for i, line in enumerate(wrap(footer, small, 940)):
            draw.text((70, 915 + i * 35), line, font=small, fill=MINT)
    canvas.save(path)


def load_inputs(summary, recording, csv_path):
    text = summary.read_text()
    lines = text.splitlines()
    if "MOCK" in text or "backend: mock" in text:
        raise ValueError("Refusing mock benchmark results.")
    header_index = next((i for i, line in enumerate(lines) if line.startswith("| system")), None)
    if header_index is None:
        raise ValueError("The benchmark summary table is missing.")
    headers = [v.strip() for v in lines[header_index].strip("|").split("|")]
    row_index = next(i for i in range(header_index + 2, len(lines)) if lines[i].startswith("| JEV"))
    cells = [v.strip() for v in lines[row_index].strip("|").split("|")]
    values = dict(zip(headers, cells))
    coverage_key = "intent_coverage@95%P"
    required = ("acc_en", "acc_uz", "acc_ru", coverage_key, "p50_ms", "usd_per_1k")
    for key in required:
        if key not in values or not math.isfinite(float(values[key])):
            raise ValueError("Missing or non-finite summary metric: " + key)
    if not all(0 <= float(values[key]) <= 1 for key in required[:4]):
        raise ValueError("Summary accuracy or coverage is outside [0, 1].")
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 36 or {r["language"] for r in rows} != {"en", "uz", "ru"}:
        raise ValueError("This film is specified for a 36-query EN/UZ/RU pilot.")
    models = {r["jev_model"] for r in rows}
    if len(models) != 1 or not all(m.startswith("jev-") for m in models):
        raise ValueError("Expected one exact Jev model version in the pilot.")
    model = next(iter(models))
    if values["system"] != f"JEV ({model})":
        raise ValueError("Summary and CSV model versions differ.")
    # Check displayed summary metrics against raw outcomes before making captions.
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from benchmark import coverage_at_precision
    conf = [float(r["jev_conf"]) for r in rows]
    correct = [r["jev_pred"] == r["gold_intent"] for r in rows]
    measured = {
        "p50_ms": float(np.median([float(r["jev_ms"]) for r in rows])),
        coverage_key: coverage_at_precision(conf, correct),
    }
    for lang in ("en", "uz", "ru"):
        group = [r for r in rows if r["language"] == lang]
        measured["acc_" + lang] = sum(r["jev_pred"] == r["gold_intent"] for r in group) / len(group)
    for key, value in measured.items():
        if not math.isclose(value, float(values[key]), abs_tol=0.00001):
            raise ValueError("Summary and CSV disagree: " + key)
    manifest = json.loads(recording.read_text())
    if not manifest.get("complete") or manifest.get("backend") not in {"typesafe", "vercel", "cloudflare"}:
        raise ValueError("A completed live recording is required.")
    if f"backend: {manifest['backend']}" not in text:
        raise ValueError("Benchmark and recording backends differ.")
    records = manifest.get("queries", [])
    if [r["query"] for r in records] != QUERIES:
        raise ValueError("The recording must contain the four requested queries in order.")
    for record in records:
        validate_decision(record["decision"], record["query"], manifest["backend"])
        if record["decision"]["model"] != model:
            raise ValueError("Recorded Jev version differs from the benchmark version.")
    raw = (recording.parent / manifest["video"]).resolve()
    if not raw.is_relative_to(recording.parent.resolve()) or not raw.is_file():
        raise ValueError("Raw video is missing or outside the recording directory.")
    warnings = []
    if all(correct):
        warnings.append("100% accuracy on this pilot. This is not evidence of generalisation.")
    if all(v == 1 for v in conf):
        warnings.append("Every intent confidence is 1.0; confidence does not separate errors here.")
    if float(values["p50_ms"]) < 10:
        warnings.append("Suspiciously low median latency, below 10 ms. Inspect the raw run.")
    if float(values["acc_uz"]) < 0.5:
        warnings.append("Uzbek accuracy is below 50%; report this limitation prominently.")
    errors = [(i + 2, r) for i, r in enumerate(rows) if r["jev_pred"] != r["gold_intent"]]
    error = max(errors, key=lambda pair: float(pair[1]["jev_conf"])) if errors else None
    return values, rows, model, records, raw, row_index + 1, warnings, error, lines


def outcome_caption(index, decision):
    route = decision["route"]
    no_llm = not decision["answer_called"] and not decision["baseline_called"]
    suffix = "No LLM call" if no_llm else "See the observed route above"
    if route == "human":
        return "Human handoff selected (demo)\n" + suffix
    if route == "human_review":
        return "Flagged for staff review (demo)\n" + suffix
    if route == "template_decline":
        return "Off-topic: template decline\n" + suffix
    count = len(decision["kept_ids"])
    noun = "passage" if count == 1 else "passages"
    if index == 0:
        urgent = "Urgency flagged" if decision["urgent"] >= 0.5 else "Urgency not flagged"
        return f"{decision['display_latency_ms']} ms · {count} {noun} kept\n{urgent}"
    if decision["answer_succeeded"]:
        return f"Jev kept {count} {noun}\nUsed as sources for the answer"
    if decision["answer_called"]:
        return f"Jev kept {count} {noun}\nAnswer generation failed"
    return f"Jev kept {count} {noun}\nAnswer generation is off"


def clip_plan(record, duration):
    start, click, result, end = (float(record[key]) for key in ("start", "click", "result", "end"))
    if not 0 <= start < click < result < end:
        raise ValueError("Invalid recording timestamps.")
    intervals = [(start, end)]
    shortened = end - start > duration
    if shortened:
        intervals = [(start, click + 0.25), (result - 0.15, end)]
    length = sum(b - a for a, b in intervals)
    if length > duration or any(b <= a for a, b in intervals):
        raise ValueError("Visible query/result content exceeds the video slot. Adjust the edit explicitly.")
    badge_at = result - start if not shortened else click + 0.25 - start + 0.15
    return intervals, max(2.5, badge_at + 0.3), shortened


def ffmpeg(args):
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        raise RuntimeError("ffmpeg failed: " + result.stderr[-2000:])


def encode_args(duration):
    return ["-t", str(duration), "-an", "-c:v", "libx264", "-preset", "medium",
            "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30", "-threads", "2"]


def fade(duration):
    return f"fade=t=in:st=0:d=0.15:color=0x003432,fade=t=out:st={duration - 0.15}:d=0.15:color=0x003432"


def encode_card(png, output, duration):
    ffmpeg(["-loop", "1", "-framerate", "30", "-i", str(png), "-vf", fade(duration),
            *encode_args(duration), str(output)])


def encode_query(raw, record, duration, intro, outcome, output, work, index):
    intervals, switch, shortened = clip_plan(record, duration)
    if duration - switch < 2.5:
        raise ValueError("The outcome caption would be visible for less than 2.5 seconds.")
    intro_path, outcome_path = work / f"intro-{index}.png", work / f"outcome-{index}.png"
    card(intro_path, "", intro, caption=True)
    card(outcome_path, "", outcome, caption=True)
    filters = []
    for i, (start, end) in enumerate(intervals):
        filters.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}]")
    labels = "".join(f"[v{i}]" for i in range(len(intervals)))
    filters.append(f"{labels}concat=n={len(intervals)}:v=1:a=0,scale=1080:1080,setsar=1,"
                   f"fps=30,tpad=stop_mode=clone:stop_duration={duration}[base]")
    filters.append(f"[base][1:v]overlay=0:0:enable='lt(t,{switch})'[intro]")
    filters.append(f"[intro][2:v]overlay=0:0:enable='gte(t,{switch})'[captioned]")
    final = "captioned"
    inputs = ["-i", str(raw), "-loop", "1", "-i", str(intro_path), "-loop", "1", "-i", str(outcome_path)]
    if shortened:
        notice = work / f"edit-notice-{index}.png"
        canvas = Image.new("RGBA", (1080, 1080), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((20, 15, 1060, 65), fill=(0, 52, 50, 235))
        draw.text((38, 24), "Waiting time shortened; badge shows measured latency", font=font(27), fill=WHITE)
        canvas.save(notice)
        inputs += ["-loop", "1", "-i", str(notice)]
        filters.append("[captioned][3:v]overlay=0:0[noticed]")
        final = "noticed"
    filters.append(f"[{final}]{fade(duration)}[out]")
    ffmpeg([*inputs, "-filter_complex", ";".join(filters), "-map", "[out]",
            *encode_args(duration), str(output)])
    return {"source_intervals": intervals, "caption_switch": switch, "wait_shortened": shortened}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, default=ROOT / "results/summary.md")
    ap.add_argument("--recording", type=Path, default=ROOT / "demo/recording.json")
    ap.add_argument("--acknowledge-pilot-warnings", action="store_true")
    args = ap.parse_args()
    for name in ("ffmpeg", "ffprobe"):
        if not shutil.which(name):
            raise ValueError(f"{name} is required. On macOS: brew install ffmpeg")
    if not args.summary.is_file() or not args.recording.is_file():
        raise ValueError("Run the real benchmark and live recorder first. No placeholder film will be produced.")
    values, rows, model, records, raw, table_line, warnings, error, summary_lines = load_inputs(
        args.summary, args.recording, args.summary.with_name("results.csv"))
    for warning in warnings:
        print("PILOT WARNING: " + warning)
    if warnings and not args.acknowledge_pilot_warnings:
        raise ValueError("Review the warnings, then rerun with --acknowledge-pilot-warnings if justified.")
    destination = args.recording.resolve().parent
    output = destination / "fastgate_demo_1080.mp4"
    if output.exists():
        raise ValueError("Final video already exists. Archive the previous version before rendering.")
    work = destination / "render"
    work.mkdir(parents=True, exist_ok=True)
    results_text = (
        "Pilot: 36 queries\n"
        f"Intent accuracy: EN {float(values['acc_en']):.1%}\n"
        f"UZ {float(values['acc_uz']):.1%} · RU {float(values['acc_ru']):.1%}\n"
        f"Intent coverage: {float(values['intent_coverage@95%P']):.1%}\n"
        "at 95% observed precision\n"
        f"p50: {float(values['p50_ms']):.0f} ms\n"
        f"Estimated $/1k: {float(values['usd_per_1k']):.6f}\n{model}"
    )
    if error:
        csv_line, row = error
        query = row["query"]
        short_query = query if len(query) <= 80 else query[:77].rstrip() + "..."
        finding = f'"{short_query}"\nExpected: {row["gold_intent"]}\nJev chose: {row["jev_pred"]}'
        error_line = next((i + 1 for i, line in enumerate(summary_lines) if query in line), None)
        if error_line is None:
            raise ValueError("Selected failure case is missing from the summary errors table.")
        finding_source = f"results/summary.md:{error_line}; results/results.csv:{csv_line}"
    else:
        finding = "No errors on this pilot set;\nscaling to 300 queries next"
        finding_source = "results/summary.md, Jev errors section; 300 is a future target, not a measured result"
    cards = [
        (0, 4, "FASTGATE", "Jev for a multilingual\nuniversity helpdesk", ""),
        (1, 4, "DECISIONS BEFORE GENERATION", "Jev decides. Code routes.\nThe LLM writes only\nwhen needed.", ""),
        (6, 10, "PILOT RESULTS", results_text,
         "Intent triage only. Coverage threshold selected on this sample.\nLatency and estimated cost exclude grounding and answer generation."),
        (7, 6, "WHERE IT STRUGGLED" if error else "NEXT VALIDATION", finding, ""),
        (8, 7, "CODE + BENCHMARK", "github.com/rajantripathi/\nfastgate-jev", "Fictional university policies. Staff handoffs are simulated."),
    ]
    ledger = []
    for index, duration, title, body, footer in cards:
        image = work / f"card-{index}.png"
        card(image, title, body, footer)
        encode_card(image, work / f"segment-{index}.mp4", duration)
        source = f"results/summary.md:{table_line}" if index == 6 else finding_source if index == 7 else "Editorial text; no measured performance claim"
        ledger.append({"segment": index, "duration": duration, "caption": body, "footer": footer, "source": source})
    intros = ["Uzbek fee question with a deadline", "Russian registration question",
              "A request to speak to staff", "An off-topic question"]
    for index, (record, duration) in enumerate(zip(records, (14, 12, 10, 8))):
        outcome = outcome_caption(index, record["decision"])
        details = encode_query(raw, record, duration, intros[index], outcome,
                               work / f"segment-{index + 2}.mp4", work, index)
        ledger.append({"segment": index + 2, "duration": duration, "intro": intros[index],
                       "caption": outcome, "source": f"demo/recording.json: queries[{index}].decision",
                       "actual_route": record["decision"]["route"], **details})
    concat = work / "segments.txt"
    # Relative generated filenames keep shell/path quoting out of the concat file.
    concat.write_text("".join(f"file 'segment-{index}.mp4'\n" for index in range(9)))
    ffmpeg(["-f", "concat", "-safe", "1", "-i", str(concat), "-c", "copy", "-an",
            "-movflags", "+faststart", str(output)])
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams",
                                               "-show_format", "-of", "json", str(output)], text=True))
    streams = probe["streams"]
    duration = float(probe["format"]["duration"])
    if (len(streams) != 1 or streams[0]["codec_name"] != "h264" or streams[0]["width"] != 1080
            or streams[0]["height"] != 1080 or streams[0]["pix_fmt"] != "yuv420p"
            or streams[0]["r_frame_rate"] != "30/1" or not 70 <= duration <= 80):
        raise ValueError("Encoded video failed the requested format checks.")
    gif = work / "demo.gif"
    for fps, colours in ((10, 96), (8, 64), (6, 48)):
        ffmpeg(["-ss", "8", "-t", "10", "-i", str(output), "-filter_complex",
                f"fps={fps},scale=720:-1:flags=lanczos,split[a][b];"
                f"[a]palettegen=max_colors={colours}:stats_mode=diff[p];[b][p]paletteuse=dither=bayer",
                "-loop", "0", str(gif)])
        if gif.stat().st_size < 5_000_000:
            break
    if gif.stat().st_size >= 5_000_000:
        raise ValueError("GIF exceeds 5 MB. The MP4 is complete; review GIF compression.")
    target_gif = ROOT / "docs/demo.gif"
    target_gif.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(gif, target_gif)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "video": str(output),
        "duration_seconds": duration, "video_bytes": output.stat().st_size,
        "gif_bytes": target_gif.stat().st_size, "model": model, "warnings": warnings,
        "inputs": {str(p): sha(p) for p in (args.summary, args.summary.with_name("results.csv"), args.recording)},
        "captions": sorted(ledger, key=lambda item: item["segment"]),
        "caption_changes": [
            "Launch-relative headline replaced with a date-independent description.",
            "Auto-routing claim replaced with intent coverage at observed precision on the same pilot.",
            "Staff handoffs labelled as simulated; no real ticket or staff connection is created.",
            "Passage counts, urgency and LLM activity follow recorded evidence for each query.",
            "Any removed waiting interval is labelled on screen; decision latency is unchanged.",
        ],
    }
    (destination / "video_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    md = ["# FastGate video evidence", "", f"- Video: {output}",
          f"- Duration: {duration:.2f} s; size: {output.stat().st_size:,} bytes",
          f"- GIF: {target_gif}; size: {target_gif.stat().st_size:,} bytes", "",
          "## Caption sources", ""]
    for entry in report["captions"]:
        md += [f"### Segment {entry['segment']}", "", entry["caption"].replace("\n", "  \n"),
               "", "Source: " + entry["source"], ""]
    md += ["## Caption changes", "", *["- " + item for item in report["caption_changes"]]]
    if warnings:
        md += ["", "## Pilot warnings", "", *["- " + item for item in warnings]]
    (destination / "video_report.md").write_text("\n".join(md) + "\n")
    print(f"Created {output} ({duration:.2f} s, {output.stat().st_size:,} bytes)")
    print(f"Created {target_gif} ({target_gif.stat().st_size:,} bytes)")
    print("Review the video and demo/video_report.md before publication.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
