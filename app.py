import os
import threading
import time
import uuid

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    send_from_directory,
    abort,
)

import engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

app = Flask(__name__)

# In-memory registry of runs: run_id -> {cfg, state, lock, thread, stop_event, dir}
RUNS = {}


def parse_user_levels(raw):
    levels = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        levels.append(int(part))
    if not levels:
        raise ValueError("Enter at least one concurrent-user value, e.g. 10,20,40,60,80,100")
    return sorted(set(levels))


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def start_run():
    form = request.form

    try:
        cfg = {
            "base_url": form.get("base_url", "http://localhost:8000").strip(),
            "model": form.get("model", "").strip(),
            "user_levels": parse_user_levels(form.get("user_levels", "10,20,40,60,80,100")),
            "min_input_tokens": int(form.get("min_input_tokens", 100)),
            "max_input_tokens": int(form.get("max_input_tokens", 7000)),
            "input_token_interval": int(form.get("input_token_interval", 1000)),
            "max_output_tokens": int(form.get("max_output_tokens", 2000)),
            "temperature": float(form.get("temperature", 0.7)),
            "per_request_timeout": int(form.get("per_request_timeout", 900)),
            "cooldown_seconds": int(form.get("cooldown_seconds", 2)),
            "base_prompt": (form.get("base_prompt") or "").strip() or engine.DEFAULT_BASE_PROMPT,
        }

        if not cfg["model"]:
            raise ValueError("Model name / path is required.")
        if cfg["min_input_tokens"] <= 0 or cfg["max_input_tokens"] < cfg["min_input_tokens"]:
            raise ValueError("Input token range is invalid.")
        if cfg["input_token_interval"] <= 0:
            raise ValueError("Input token interval must be positive.")
        if cfg["max_output_tokens"] <= 0:
            raise ValueError("Max output tokens must be positive.")
        if any(u <= 0 for u in cfg["user_levels"]):
            raise ValueError("Concurrent-user values must be positive integers.")

    except ValueError as e:
        return render_template("index.html", error=str(e), form=form), 400

    run_id = uuid.uuid4().hex[:10]
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    input_levels = list(range(
        cfg["min_input_tokens"], cfg["max_input_tokens"] + 1, cfg["input_token_interval"]
    ))
    total_rounds = len(cfg["user_levels"]) * len(input_levels)
    total_requests = sum(u for u in cfg["user_levels"]) * len(input_levels)

    state = {
        "running": False,
        "done": False,
        "error": None,
        "current_round": 0,
        "total_rounds": total_rounds,
        "total_requests": total_requests,
        "summaries": [],
        "log": [],
        "started_at": None,
        "gpu_before": None,
        "gpu_after": None,
        "chart_files": [],
    }

    lock = threading.Lock()
    stop_event = threading.Event()

    thread = threading.Thread(
        target=engine.run_full_test,
        args=(cfg, run_dir, state, lock, stop_event),
        daemon=True,
    )

    RUNS[run_id] = {
        "cfg": cfg,
        "state": state,
        "lock": lock,
        "thread": thread,
        "stop_event": stop_event,
        "dir": run_dir,
    }

    thread.start()

    return redirect(url_for("dashboard", run_id=run_id))


@app.route("/dashboard/<run_id>")
def dashboard(run_id):
    run = RUNS.get(run_id)
    if run is None:
        abort(404)
    return render_template("dashboard.html", run_id=run_id, cfg=run["cfg"])


@app.route("/api/status/<run_id>")
def api_status(run_id):
    run = RUNS.get(run_id)
    if run is None:
        return jsonify({"error": "unknown run_id"}), 404

    with run["lock"]:
        state = dict(run["state"])
        state["summaries"] = list(state["summaries"])
        state["log"] = list(state["log"])

    return jsonify(state)


@app.route("/stop/<run_id>", methods=["POST"])
def stop_run(run_id):
    run = RUNS.get(run_id)
    if run is None:
        return jsonify({"error": "unknown run_id"}), 404
    run["stop_event"].set()
    return jsonify({"stopping": True})


@app.route("/download/<run_id>/pdf")
def download_pdf(run_id):
    run = RUNS.get(run_id)
    if run is None:
        abort(404)
    path = os.path.join(run["dir"], "report.pdf")
    if not os.path.exists(path):
        abort(404)
    return send_from_directory(run["dir"], "report.pdf", as_attachment=True,
                                download_name=f"load_test_report_{run_id}.pdf")


@app.route("/download/<run_id>/json")
def download_json(run_id):
    run = RUNS.get(run_id)
    if run is None:
        abort(404)
    path = os.path.join(run["dir"], "results.json")
    if not os.path.exists(path):
        abort(404)
    return send_from_directory(run["dir"], "results.json", as_attachment=True,
                                download_name=f"load_test_results_{run_id}.json")


@app.route("/charts/<run_id>/<filename>")
def chart_file(run_id, filename):
    run = RUNS.get(run_id)
    if run is None:
        abort(404)
    chart_dir = os.path.join(run["dir"], "charts")
    return send_from_directory(chart_dir, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
