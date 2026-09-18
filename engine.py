import json
import os
import statistics
import subprocess
import threading
import time
import urllib.request
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)


DEFAULT_BASE_PROMPT = """
Write a complete, production-quality implementation of a red-black tree
in Python, including insert, delete, and rebalancing logic. Explain each
rotation case with inline comments, then write unit tests covering every
edge case, then explain the time complexity of each operation in detail.

The following context is additional input data. Carefully read it before
answering. Do not ignore it.
""".strip()

FILLER = (
    "This is additional test context used for measuring LLM inference "
    "performance under different input lengths. The information is "
    "intentionally repetitive and should be considered part of the user "
    "input. Analyze the complete request before generating the answer. "
)


# ===========================================================================
# PROMPT GENERATION
# ===========================================================================

def build_prompt(base_prompt, target_tokens):
    target_chars = target_tokens * 4

    if target_chars <= len(base_prompt):
        return base_prompt[:target_chars]

    remaining_chars = target_chars - len(base_prompt) - 2
    repetitions = (remaining_chars // len(FILLER)) + 1
    filler = (FILLER * repetitions)[:remaining_chars]

    return base_prompt + "\n\n" + filler


# ===========================================================================
# REQUEST HELPER
# ===========================================================================

def send_request(cfg, result_slot, idx, input_tokens):
    url = f"{cfg['base_url'].rstrip('/')}/v1/chat/completions"

    prompt = build_prompt(cfg["base_prompt"], input_tokens)

    body = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": cfg["max_output_tokens"],
        "temperature": cfg["temperature"],
        "stream": False,
    }

    request_data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=request_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start_time = time.time()

    try:
        with urllib.request.urlopen(req, timeout=cfg["per_request_timeout"]) as response:
            data = json.load(response)

        latency = time.time() - start_time

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens

        request_output_tps = (completion_tokens / latency) if latency > 0 else 0

        result_slot[idx] = {
            "ok": True,
            "latency": latency,
            "requested_input_tokens": input_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "request_output_tps": request_output_tps,
            "error": None,
        }

    except Exception as e:
        latency = time.time() - start_time

        result_slot[idx] = {
            "ok": False,
            "latency": latency,
            "requested_input_tokens": input_tokens,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "request_output_tps": 0,
            "error": str(e)[:500],
        }


# ===========================================================================
# LOAD TEST (ONE ROUND)
# ===========================================================================

def run_load_test(cfg, round_number, input_tokens, n_users, log=None):
    def emit(msg):
        if log is not None:
            log.append(msg)

    emit(f"Round {round_number}: {n_users} users x {input_tokens} input tokens - starting")

    results = [None] * n_users
    threads = []

    for i in range(n_users):
        t = threading.Thread(target=send_request, args=(cfg, results, i, input_tokens))
        threads.append(t)

    wall_start = time.time()

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    total_wall_time = time.time() - wall_start

    successes = [r for r in results if r is not None and r["ok"]]
    failures = [r for r in results if r is not None and not r["ok"]]

    latencies = sorted(r["latency"] for r in successes)

    total_prompt_tokens = sum(r["prompt_tokens"] for r in successes)
    total_completion_tokens = sum(r["completion_tokens"] for r in successes)
    total_tokens = total_prompt_tokens + total_completion_tokens

    def percentile(p):
        if not latencies:
            return None
        position = (p / 100) * (len(latencies) - 1)
        index = int(round(position))
        return latencies[index]

    average_prompt_tokens = (
        statistics.mean(r["prompt_tokens"] for r in successes) if successes else None
    )

    aggregate_output_tps = (total_completion_tokens / total_wall_time) if total_wall_time > 0 else 0
    aggregate_input_tps = (total_prompt_tokens / total_wall_time) if total_wall_time > 0 else 0

    output_tps_per_user = (aggregate_output_tps / n_users) if n_users > 0 else 0

    avg_per_request_output_tps = (
        statistics.mean(r["request_output_tps"] for r in successes) if successes else 0
    )

    summary = {
        "round": round_number,
        "users": n_users,
        "input_tokens_requested": input_tokens,
        "total_wall_time": total_wall_time,
        "success_count": len(successes),
        "failure_count": len(failures),
        "success_rate": (len(successes) / n_users) if n_users > 0 else 0,
        "average_prompt_tokens": average_prompt_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "aggregate_input_tps": aggregate_input_tps,
        "aggregate_tps": aggregate_output_tps,
        "output_tps_per_user": output_tps_per_user,
        "avg_per_request_output_tps": avg_per_request_output_tps,
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "min_latency": min(latencies) if latencies else None,
        "max_latency": max(latencies) if latencies else None,
        "avg_latency": statistics.mean(latencies) if latencies else None,
        "sample_errors": list({r["error"] for r in failures if r["error"]})[:5],
    }

    emit(
        f"Round {round_number}: done - "
        f"{summary['success_count']}/{n_users} ok "
        f"({summary['success_rate']*100:.0f}%), "
        f"out TPS={aggregate_output_tps:.1f}, "
        f"TPS/user={output_tps_per_user:.2f}"
        + (f", p50={summary['p50']:.2f}s" if summary["p50"] is not None else "")
    )

    if failures:
        for err in summary["sample_errors"]:
            emit(f"  error: {err}")

    return summary


# ===========================================================================
# GPU SNAPSHOT
# ===========================================================================

def get_gpu_snapshot():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            timeout=5,
        ).decode().strip()
        return output
    except Exception:
        return None


# ===========================================================================
# CHART HELPERS
# ===========================================================================

def _line_chart_by_users(summaries, user_levels, y_key, y_label, title, path, y_percent=False):
    fig, ax = plt.subplots(figsize=(11, 6))

    for users in user_levels:
        rows = [s for s in summaries if s["users"] == users]
        rows.sort(key=lambda x: x["input_tokens_requested"])

        xs = [r["input_tokens_requested"] for r in rows]
        ys = [(r[y_key] or 0) * (100 if y_percent else 1) for r in rows]

        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.5, label=f"{users} users")

    ax.set_xlabel("Input tokens")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(title="Concurrent users", ncol=2)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return path


def _line_chart_by_input_tokens(summaries, y_key, y_label, title, path, y_percent=False):
    input_levels = sorted({s["input_tokens_requested"] for s in summaries})

    fig, ax = plt.subplots(figsize=(11, 6))

    for input_tokens in input_levels:
        rows = [s for s in summaries if s["input_tokens_requested"] == input_tokens]
        rows.sort(key=lambda x: x["users"])

        xs = [r["users"] for r in rows]
        ys = [(r[y_key] or 0) * (100 if y_percent else 1) for r in rows]

        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.5, label=f"{input_tokens} input tok")

    ax.set_xlabel("Concurrent users")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(title="Input tokens", ncol=2, fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return path


def build_all_charts(summaries, user_levels, chart_dir):
    os.makedirs(chart_dir, exist_ok=True)

    charts = []

    charts.append((
        "Aggregate Output Throughput vs Input Tokens",
        _line_chart_by_users(
            summaries, user_levels, "aggregate_tps",
            "Aggregate output tokens / second",
            "Aggregate Output Throughput vs Input Tokens",
            os.path.join(chart_dir, "01_throughput_vs_input_tokens.png"),
        ),
    ))

    charts.append((
        "Input Processing Throughput vs Input Tokens",
        _line_chart_by_users(
            summaries, user_levels, "aggregate_input_tps",
            "Aggregate input tokens / second",
            "Input Processing Throughput vs Input Tokens",
            os.path.join(chart_dir, "02_input_throughput_vs_input_tokens.png"),
        ),
    ))

    charts.append((
        "p50 Latency vs Input Tokens",
        _line_chart_by_users(
            summaries, user_levels, "p50",
            "p50 latency (seconds)",
            "p50 Latency vs Input Tokens",
            os.path.join(chart_dir, "03_latency_p50_vs_input_tokens.png"),
        ),
    ))

    charts.append((
        "Request Success Rate vs Input Tokens",
        _line_chart_by_users(
            summaries, user_levels, "success_rate",
            "Success rate (%)",
            "Request Success Rate vs Input Tokens",
            os.path.join(chart_dir, "04_success_rate_vs_input_tokens.png"),
            y_percent=True,
        ),
    ))

    charts.append((
        "Output Throughput Per User vs Input Tokens",
        _line_chart_by_users(
            summaries, user_levels, "output_tps_per_user",
            "Output tokens / second / user",
            "Output Throughput Per User vs Input Tokens",
            os.path.join(chart_dir, "05_output_tps_per_user_vs_input_tokens.png"),
        ),
    ))

    charts.append((
        "Average Per-Request Output Speed vs Input Tokens",
        _line_chart_by_users(
            summaries, user_levels, "avg_per_request_output_tps",
            "Avg per-request output tokens / second",
            "Average Per-Request Output Speed vs Input Tokens",
            os.path.join(chart_dir, "06_avg_per_request_tps_vs_input_tokens.png"),
        ),
    ))

    charts.append((
        "p90 Latency vs Input Tokens",
        _line_chart_by_users(
            summaries, user_levels, "p90",
            "p90 latency (seconds)",
            "p90 Latency vs Input Tokens",
            os.path.join(chart_dir, "07_latency_p90_vs_input_tokens.png"),
        ),
    ))

    charts.append((
        "p99 Latency vs Input Tokens",
        _line_chart_by_users(
            summaries, user_levels, "p99",
            "p99 latency (seconds)",
            "p99 Latency vs Input Tokens",
            os.path.join(chart_dir, "08_latency_p99_vs_input_tokens.png"),
        ),
    ))

    charts.append((
        "Aggregate Output Throughput vs Concurrent Users",
        _line_chart_by_input_tokens(
            summaries, "aggregate_tps",
            "Aggregate output tokens / second",
            "Aggregate Output Throughput vs Concurrent Users",
            os.path.join(chart_dir, "09_throughput_vs_users.png"),
        ),
    ))

    charts.append((
        "p50 Latency vs Concurrent Users",
        _line_chart_by_input_tokens(
            summaries, "p50",
            "p50 latency (seconds)",
            "p50 Latency vs Concurrent Users",
            os.path.join(chart_dir, "10_latency_p50_vs_users.png"),
        ),
    ))

    return charts


# ===========================================================================
# PDF REPORT
# ===========================================================================

def build_pdf(cfg, summaries, user_levels, gpu_before, gpu_after, chart_dir, pdf_path):
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=landscape(letter),
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
        leftMargin=0.4 * inch,
        rightMargin=0.4 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    h2 = styles["Heading2"]
    normal = styles["Normal"]
    small = ParagraphStyle("small", parent=normal, fontSize=7, textColor=colors.grey)

    story = []

    story.append(Paragraph("LLM Load Test Report", title_style))
    story.append(Paragraph(f"<b>Model:</b> {cfg['model']}", normal))
    story.append(Paragraph(f"<b>Server:</b> {cfg['base_url']}", normal))
    story.append(Paragraph(f"<b>Total rounds:</b> {len(summaries)}", normal))
    story.append(Paragraph(
        f"<b>Input token range:</b> {cfg['min_input_tokens']} - "
        f"{cfg['max_input_tokens']} (step {cfg['input_token_interval']})",
        normal,
    ))
    story.append(Paragraph(f"<b>User levels:</b> {user_levels}", normal))
    story.append(Paragraph(f"<b>Max output tokens:</b> {cfg['max_output_tokens']}", normal))
    story.append(Paragraph(f"<b>Temperature:</b> {cfg['temperature']}", normal))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal))

    if gpu_before:
        story.append(Paragraph(f"GPU before: {gpu_before}", small))
    if gpu_after:
        story.append(Paragraph(f"GPU after: {gpu_after}", small))

    story.append(Spacer(1, 10))

    charts = build_all_charts(summaries, user_levels, chart_dir)

    for heading, chart_path in charts:
        story.append(Paragraph(heading, h2))
        story.append(Image(chart_path, width=9.7 * inch, height=5.1 * inch))
        story.append(PageBreak())

    story.append(Paragraph("Detailed Results", h2))

    rows = [[
        "Round", "Users", "Input", "OK", "Fail", "Success %",
        "Actual Input", "Output", "Input TPS", "Output TPS",
        "TPS/User", "p50", "p95", "p99", "Max",
    ]]

    for s in summaries:
        rows.append([
            str(s["round"]),
            str(s["users"]),
            str(s["input_tokens_requested"]),
            str(s["success_count"]),
            str(s["failure_count"]),
            f"{s['success_rate'] * 100:.1f}%",
            (f"{s['average_prompt_tokens']:.0f}" if s["average_prompt_tokens"] is not None else "N/A"),
            str(s["total_completion_tokens"]),
            f"{s['aggregate_input_tps']:.1f}",
            f"{s['aggregate_tps']:.1f}",
            f"{s['output_tps_per_user']:.1f}",
            (f"{s['p50']:.2f}" if s["p50"] is not None else "N/A"),
            (f"{s['p95']:.2f}" if s["p95"] is not None else "N/A"),
            (f"{s['p99']:.2f}" if s["p99"] is not None else "N/A"),
            (f"{s['max_latency']:.2f}" if s["max_latency"] is not None else "N/A"),
        ])

    col_widths = [32, 35, 42, 30, 30, 48, 56, 48, 52, 52, 48, 38, 38, 38, 38]

    table = Table(rows, colWidths=col_widths, repeatRows=1)

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C72B0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 6.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    story.append(table)
    story.append(Spacer(1, 10))

    any_errors = any(s["failure_count"] > 0 for s in summaries)

    if any_errors:
        story.append(Paragraph("Sample Errors", h2))

        for s in summaries:
            if not s["sample_errors"]:
                continue

            story.append(Paragraph(
                f"<b>Round {s['round']}: {s['input_tokens_requested']} input tokens / "
                f"{s['users']} users</b>",
                normal,
            ))

            for err in s["sample_errors"]:
                story.append(Paragraph(err, small))

            story.append(Spacer(1, 5))

    doc.build(story)

    return pdf_path


# ===========================================================================
# FULL RUN ORCHESTRATION (used by the Flask app)
# ===========================================================================

def run_full_test(cfg, run_dir, state, lock, stop_event):
    """
    Runs every round (user_level x input_token_level), updating `state`
    (a dict guarded by `lock`) after every round so a web UI can poll
    progress. Writes results.json, report.pdf and chart PNGs into
    run_dir when finished (or when stopped early).
    """

    os.makedirs(run_dir, exist_ok=True)
    chart_dir = os.path.join(run_dir, "charts")
    os.makedirs(chart_dir, exist_ok=True)

    input_token_levels = list(range(
        cfg["min_input_tokens"],
        cfg["max_input_tokens"] + 1,
        cfg["input_token_interval"],
    ))

    user_levels = cfg["user_levels"]

    rounds = []
    round_number = 0
    for n_users in user_levels:
        for input_tokens in input_token_levels:
            round_number += 1
            rounds.append({"round": round_number, "users": n_users, "input_tokens": input_tokens})

    with lock:
        state["total_rounds"] = len(rounds)
        state["current_round"] = 0
        state["running"] = True
        state["done"] = False
        state["error"] = None
        state["summaries"] = []
        state["log"] = []
        state["started_at"] = time.time()

    gpu_before = get_gpu_snapshot()

    summaries = []

    try:
        for test in rounds:
            if stop_event.is_set():
                with lock:
                    state["log"].append("Stop requested - halting after current round.")
                break

            summary = run_load_test(
                cfg,
                round_number=test["round"],
                input_tokens=test["input_tokens"],
                n_users=test["users"],
                log=None,
            )

            summaries.append(summary)

            with lock:
                state["current_round"] = test["round"]
                state["summaries"] = list(summaries)
                state["log"].append(
                    f"Round {summary['round']}/{len(rounds)}: "
                    f"{summary['users']} users x {summary['input_tokens_requested']} tok - "
                    f"{summary['success_count']}/{summary['users']} ok, "
                    f"out TPS={summary['aggregate_tps']:.1f}, "
                    f"TPS/user={summary['output_tps_per_user']:.2f}"
                    + (f", p50={summary['p50']:.2f}s" if summary["p50"] is not None else "")
                )
                if len(state["log"]) > 500:
                    state["log"] = state["log"][-500:]

            with open(os.path.join(run_dir, "results.json"), "w") as f:
                json.dump(summaries, f, indent=2)

            if test["round"] < len(rounds) and not stop_event.is_set():
                time.sleep(cfg["cooldown_seconds"])

        gpu_after = get_gpu_snapshot()

        pdf_path = os.path.join(run_dir, "report.pdf")

        if summaries:
            build_pdf(cfg, summaries, user_levels, gpu_before, gpu_after, chart_dir, pdf_path)
            charts = build_all_charts(summaries, user_levels, chart_dir)
        else:
            charts = []

        with lock:
            state["running"] = False
            state["done"] = True
            state["gpu_before"] = gpu_before
            state["gpu_after"] = gpu_after
            state["chart_files"] = [os.path.basename(c[1]) for c in charts]
            state["log"].append("Test complete. Report and charts ready.")

    except Exception as e:
        with lock:
            state["running"] = False
            state["done"] = True
            state["error"] = str(e)
            state["log"].append(f"FATAL ERROR: {e}")
