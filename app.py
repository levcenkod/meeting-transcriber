#!/usr/bin/env python3
"""
app.py — Flask web UI for Meeting Transcriber.
Runs inside the Docker container; orchestrates the transcription pipeline
(WhisperX → postprocess → LLM summary) as subprocesses.
"""

import json
import os
import queue
import re
import subprocess
import threading
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file

app = Flask(__name__, template_folder="/app/templates")

INPUT_DIR  = Path("/input")
OUTPUT_DIR = Path("/output")

_SUPPORTED_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".mp4", ".mkv", ".flac", ".aac", ".webm"}

# job_id -> {status, log_queue, result}
_jobs: dict[str, dict] = {}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_category(filename: str) -> str:
    name = filename.lower()
    if "logistics" in name: return "Логистика"
    if "marketing" in name: return "Маркетинг"
    if "finance"   in name: return "Финансы"
    return "Общее"


def _log(q: queue.Queue, text: str, level: str = "info") -> None:
    q.put({"type": "log", "level": level, "text": text})
    print(f"[{level.upper()}] {text}", flush=True)


def _run_step(q: queue.Queue, cmd: list, env: dict, label: str) -> bool:
    """Run a subprocess, stream every output line to the SSE queue. Returns True on success."""
    _log(q, f"▶ {label}", "step")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                _log(q, line)
        proc.wait()
        if proc.returncode != 0:
            _log(q, f"✗ {label} завершился с ошибкой (exit {proc.returncode})", "error")
            return False
        _log(q, f"✓ {label}", "ok")
        return True
    except Exception as exc:
        _log(q, f"✗ {label}: {exc}", "error")
        return False


def _pipeline(job_id: str, audio_path: Path, category: str, language: str, llm_enabled: bool) -> None:
    """Main pipeline — runs in a background daemon thread."""
    job = _jobs[job_id]
    q: queue.Queue = job["log_queue"]
    stem    = audio_path.stem
    out_dir = OUTPUT_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)
    job["status"] = "running"

    try:
        # ── Build environment ────────────────────────────────────────────────
        env = {**os.environ}
        env["OUTPUT_SUBDIR"] = category
        env["LANGUAGE"]      = language

        # Rewrite localhost → host.docker.internal so summarize.py can reach
        # an LLM server running on the host machine
        llm_url = env.get("LLM_BASE_URL", "")
        if llm_url:
            env["LLM_BASE_URL"] = re.sub(r"localhost|127\.0\.0\.1", "host.docker.internal", llm_url)

        # ── Step 1: WhisperX transcription ───────────────────────────────────
        ok = _run_step(q, ["bash", "/scripts/entrypoint.sh", str(audio_path)], env,
                       "WhisperX транскрипция")
        if not ok:
            job["status"] = "error"
            return

        # ── Step 2: Postprocess → _speakers.txt ─────────────────────────────
        ok = _run_step(q, [
            "python", "/scripts/postprocess.py",
            "--output-dir", str(out_dir),
            "--filename",   stem,
        ], env, "Постобработка")
        if not ok:
            _log(q, "Постобработка завершилась с предупреждением, продолжаем...", "warn")

        # ── Step 3: LLM summary (optional) ──────────────────────────────────
        speakers_file = out_dir / f"{stem}_speakers.txt"
        if llm_enabled and env.get("LLM_BASE_URL") and speakers_file.exists():
            _run_step(q, [
                "python", "/scripts/summarize.py",
                "--speakers-file", str(speakers_file),
            ], env, "LLM анализ")
        elif llm_enabled and not env.get("LLM_BASE_URL"):
            _log(q, "LLM_BASE_URL не задан в .env — анализ пропущен", "warn")

        # ── Collect result files ─────────────────────────────────────────────
        _PRIMARY = {f"{stem}_summary.md", f"{stem}_speakers.txt", f"{stem}_anonymized_speakers.txt"}
        primary, secondary = [], []
        for f in sorted(out_dir.glob(f"{stem}*")):
            if not f.is_file() or "intermediate" in f.parts:
                continue
            entry = {"name": f.name, "path": str(f.relative_to(OUTPUT_DIR)), "size": f.stat().st_size}
            (primary if f.name in _PRIMARY else secondary).append(entry)

        summary_path = out_dir / f"{stem}_summary.md"
        summary_md   = summary_path.read_text(encoding="utf-8") if summary_path.exists() else None

        job["result"] = {
            "primary": primary, "secondary": secondary,
            "summary_md": summary_md, "category": category,
        }
        job["status"] = "done"
        _log(q, f"✅ Готово! Файлы сохранены в output/{category}/", "ok")

    except Exception as exc:
        _log(q, f"Неожиданная ошибка: {exc}", "error")
        job["status"] = "error"
    finally:
        q.put(None)  # Signal end-of-stream to SSE generator


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if any(j["status"] == "running" for j in _jobs.values()):
        return jsonify({"error": "Уже выполняется транскрипция. Дождитесь завершения."}), 409

    f = request.files.get("audio")
    if not f or not f.filename:
        return jsonify({"error": "Файл не выбран"}), 400

    filename = Path(f.filename).name          # strip directory components
    ext      = Path(filename).suffix.lower()
    if ext not in _SUPPORTED_EXTS:
        return jsonify({"error": f"Неподдерживаемый формат: {ext}"}), 400

    INPUT_DIR.mkdir(exist_ok=True)
    audio_path = INPUT_DIR / filename
    f.save(str(audio_path))

    category    = (request.form.get("category") or "").strip() or _get_category(filename)
    language    = request.form.get("language", "ru")
    llm_enabled = request.form.get("llm_enabled", "true") == "true"

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "pending", "log_queue": queue.Queue(), "result": None}

    threading.Thread(
        target=_pipeline,
        args=(job_id, audio_path, category, language, llm_enabled),
        daemon=True,
    ).start()

    return jsonify({"job_id": job_id, "filename": filename, "category": category})


@app.route("/stream/<job_id>")
def stream(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    def generate():
        q = job["log_queue"]
        while True:
            try:
                msg = q.get(timeout=30)
                if msg is None:
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    break
                yield f"data: {json.dumps(msg)}\n\n"
            except queue.Empty:
                # keepalive ping so the browser doesn't close the connection
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/result/<job_id>")
def result(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"status": job["status"], "result": job.get("result")})


@app.route("/download/<path:rel_path>")
def download(rel_path: str):
    full = (OUTPUT_DIR / rel_path).resolve()
    # Path traversal guard
    if not str(full).startswith(str(OUTPUT_DIR.resolve())):
        return jsonify({"error": "Forbidden"}), 403
    if not full.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(str(full), as_attachment=True)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    INPUT_DIR.mkdir(exist_ok=True)
    app.run(host="0.0.0.0", port=8080, threaded=True)
