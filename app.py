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
import time
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file

app = Flask(__name__, template_folder="/app/templates")

INPUT_DIR  = Path("/input")
OUTPUT_DIR = Path("/output")
OBSIDIAN_DIR = OUTPUT_DIR / "_obsidian"

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


_FS_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _slugify_filename(name: str, max_len: int = 80) -> str:
    """Make a string safe for use as a filename across OSes."""
    s = _FS_UNSAFE.sub("-", name).strip().strip(".")
    s = re.sub(r"\s+", " ", s)
    if len(s) > max_len:
        s = s[:max_len].rstrip()
    return s or "meeting"


def _create_obsidian_note(out_dir: Path, stem: str) -> Path | None:
    """Create a nicely-named Obsidian-friendly copy of the summary.

    Reads {stem}_summary.md (already includes YAML frontmatter) and
    {stem}_meta.json (for title/date/attendees) and writes it under
    OUTPUT_DIR/_obsidian/{Category}/{YYYY-MM-DD} — {Title}.md
    with wiki-links wrapped around known attendees.
    """
    meta_path    = out_dir / f"{stem}_meta.json"
    summary_path = out_dir / f"{stem}_summary.md"
    if not (meta_path.exists() and summary_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    title     = meta.get("title", "Совещание")
    date      = meta.get("date", "")
    category  = meta.get("category") or out_dir.name
    attendees = [a for a in meta.get("attendees", []) if a and not a.startswith("SPEAKER_")]

    content = summary_path.read_text(encoding="utf-8")

    # Wrap attendee names in [[wiki-links]] inside the body
    # (only those that are real names, longest first to avoid partial matches)
    if attendees:
        body_start = content.find("\n\n", content.find("---\n", 4) + 4) if content.startswith("---") else 0
        head, body = content[:body_start], content[body_start:]
        for name in sorted(attendees, key=len, reverse=True):
            # Skip if already wiki-linked
            pattern = re.compile(rf"(?<!\[)\b{re.escape(name)}\b(?!\]\])")
            body = pattern.sub(f"[[{name}]]", body)
        content = head + body

    folder = OBSIDIAN_DIR / _slugify_filename(category, 40)
    folder.mkdir(parents=True, exist_ok=True)
    fname = f"{date} — {_slugify_filename(title)}.md" if date else f"{_slugify_filename(title)}.md"
    dest = folder / fname
    dest.write_text(content, encoding="utf-8")
    return dest


def _read_meta(meta_path: Path) -> dict | None:
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _log(q_or_job, text: str, level: str = "info") -> None:
    """Append a log entry to the job's buffer and fan it out to all subscribers."""
    msg = {"type": "log", "level": level, "text": text}
    # Support both legacy (queue) and new (job dict) call styles
    if isinstance(q_or_job, dict):
        job = q_or_job
        job.setdefault("log_buffer", []).append(msg)
        # Cap buffer at 2000 entries
        if len(job["log_buffer"]) > 2000:
            del job["log_buffer"][:len(job["log_buffer"]) - 2000]
        for sub in list(job.get("subscribers", [])):
            try:
                sub.put_nowait(msg)
            except Exception:
                pass
    else:
        # Legacy queue-only call — kept for compatibility
        q_or_job.put(msg)
    print(f"[{level.upper()}] {text}", flush=True)


def _run_step(job: dict, cmd: list, env: dict, label: str) -> bool:
    """Run a subprocess, stream every output line. Returns True on success."""
    _log(job, f"▶ {label}", "step")
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
                _log(job, line)
        proc.wait()
        if proc.returncode != 0:
            _log(job, f"✗ {label} завершился с ошибкой (exit {proc.returncode})", "error")
            return False
        _log(job, f"✓ {label}", "ok")
        return True
    except Exception as exc:
        _log(job, f"✗ {label}: {exc}", "error")
        return False


def _pipeline(job_id: str, audio_path: Path, category: str, language: str, llm_enabled: bool) -> None:
    """Main pipeline — runs in a background daemon thread."""
    job = _jobs[job_id]
    stem    = audio_path.stem
    out_dir = OUTPUT_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)
    job["status"] = "running"
    job["out_dir"] = out_dir
    job["stem"]    = stem

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
        ok = _run_step(job, ["bash", "/scripts/entrypoint.sh", str(audio_path)], env,
                       "WhisperX транскрипция")
        if not ok:
            job["status"] = "error"
            return

        # ── Step 2: Postprocess → _speakers.txt ─────────────────────────────
        ok = _run_step(job, [
            "python", "/scripts/postprocess.py",
            "--output-dir", str(out_dir),
            "--filename",   stem,
        ], env, "Постобработка")
        if not ok:
            _log(job, "Постобработка завершилась с предупреждением, продолжаем...", "warn")

        # ── Step 3: LLM summary (optional) ──────────────────────────────────
        speakers_file = out_dir / f"{stem}_speakers.txt"
        if llm_enabled and env.get("LLM_BASE_URL") and speakers_file.exists():
            _run_step(job, [
                "python", "/scripts/summarize.py",
                "--speakers-file", str(speakers_file),
            ], env, "LLM анализ")
        elif llm_enabled and not env.get("LLM_BASE_URL"):
            _log(job, "LLM_BASE_URL не задан в .env — анализ пропущен", "warn")

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
        # ── Create Obsidian-friendly copy (if meta available) ────────────────
        try:
            obs = _create_obsidian_note(out_dir, stem)
            if obs:
                _log(job, f"📒 Obsidian-заметка: {obs.relative_to(OUTPUT_DIR)}", "ok")
        except Exception as e:
            _log(job, f"Не удалось создать Obsidian-заметку: {e}", "warn")

        job["status"] = "done"
        _log(job, f"✅ Готово! Файлы сохранены в output/{category}/", "ok")

    except Exception as exc:
        _log(job, f"Неожиданная ошибка: {exc}", "error")
        job["status"] = "error"
    finally:
        # Notify all subscribers that the stream is done
        for sub in list(job.get("subscribers", [])):
            try:
                sub.put_nowait(None)
            except Exception:
                pass
        job["finished_at"] = time.time()


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
    _jobs[job_id] = {
        "status":      "pending",
        "result":      None,
        "filename":    filename,
        "category":    category,
        "started_at":  time.time(),
        "finished_at": None,
        "log_buffer":  [],     # list of message dicts (replayed on (re)connect)
        "subscribers": [],     # list of queue.Queue, one per active SSE client
    }

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
        # Subscribe with a fresh queue
        sub = queue.Queue()
        job.setdefault("subscribers", []).append(sub)

        try:
            # Replay buffered messages so a (re)connecting client gets full history
            for msg in list(job.get("log_buffer", [])):
                yield f"data: {json.dumps(msg)}\n\n"

            # If job already finished before/while replaying, send done and exit
            if job["status"] in ("done", "error"):
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            # Live tail
            while True:
                try:
                    msg = sub.get(timeout=30)
                    if msg is None:
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                        break
                    yield f"data: {json.dumps(msg)}\n\n"
                except queue.Empty:
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"
        finally:
            try:
                job["subscribers"].remove(sub)
            except (ValueError, KeyError):
                pass

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/jobs")
def list_jobs():
    """Return all jobs known to this server (running + recently finished, in-memory)."""
    items = []
    for jid, j in _jobs.items():
        items.append({
            "job_id":      jid,
            "status":      j.get("status"),
            "filename":    j.get("filename"),
            "category":    j.get("category"),
            "started_at":  j.get("started_at"),
            "finished_at": j.get("finished_at"),
            "stem":        j.get("stem"),
        })
    # newest first
    items.sort(key=lambda x: x.get("started_at") or 0, reverse=True)
    return jsonify({"jobs": items})


@app.route("/result/<job_id>")
def result(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    return jsonify({
        "status":   job["status"],
        "result":   job.get("result"),
        "filename": job.get("filename"),
        "category": job.get("category"),
    })


def _parse_speakers(speakers_txt: Path) -> list:
    """Return [{code, samples}] from a *_speakers.txt file."""
    text    = speakers_txt.read_text(encoding="utf-8")
    pattern = re.compile(r'\[\d+:\d+:\d+\s*-\s*\d+:\d+:\d+\]\s*(SPEAKER_\w+):')
    speakers: dict[str, list] = {}
    current  = None
    for line in text.splitlines():
        m = pattern.match(line.strip())
        if m:
            current = m.group(1)
            speakers.setdefault(current, [])
        elif current and line.strip() and len(speakers[current]) < 3:
            speakers[current].append(line.strip())
    return [{"code": k, "samples": v} for k, v in sorted(speakers.items())]


def _refresh_result(job: dict) -> None:
    """Rebuild job['result'] file lists and summary from disk."""
    out_dir = job["out_dir"]
    stem    = job["stem"]
    category = job.get("result", {}).get("category", out_dir.name)
    _PRIMARY = {f"{stem}_summary.md", f"{stem}_speakers.txt", f"{stem}_anonymized_speakers.txt"}
    primary, secondary = [], []
    for f in sorted(out_dir.glob(f"{stem}*")):
        if not f.is_file() or "intermediate" in f.parts:
            continue
        entry = {"name": f.name, "path": str(f.relative_to(OUTPUT_DIR)), "size": f.stat().st_size}
        (primary if f.name in _PRIMARY else secondary).append(entry)
    summary_path = out_dir / f"{stem}_summary.md"
    summary_md   = summary_path.read_text(encoding="utf-8") if summary_path.exists() else None
    job["result"] = {"primary": primary, "secondary": secondary,
                     "summary_md": summary_md, "category": category}


@app.route("/speakers/<job_id>")
def speakers_info(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Not found"}), 404
    out_dir = job.get("out_dir")
    stem    = job.get("stem")
    if not out_dir or not stem:
        return jsonify({"speakers": []})
    sp_path = out_dir / f"{stem}_speakers.txt"
    if not sp_path.exists():
        return jsonify({"speakers": []})
    return jsonify({"speakers": _parse_speakers(sp_path)})


@app.route("/rename-speakers/<job_id>", methods=["POST"])
def rename_speakers(job_id: str):
    job = _jobs.get(job_id)
    if not job or job.get("out_dir") is None:
        return jsonify({"error": "Job not found or not ready"}), 400
    mapping = request.get_json(force=True)
    if not isinstance(mapping, dict):
        return jsonify({"error": "Expected JSON object"}), 400
    # Filter out empty names
    mapping = {k: v for k, v in mapping.items() if isinstance(v, str) and v.strip()}
    if not mapping:
        return jsonify({"error": "No names provided"}), 400
    out_dir = job["out_dir"]
    stem    = job["stem"]
    text_exts = {".txt", ".md", ".srt", ".vtt", ".tsv", ".json"}
    for f in out_dir.glob(f"{stem}*"):
        if not f.is_file() or f.suffix.lower() not in text_exts:
            continue
        if "intermediate" in f.parts:
            continue
        try:
            content = f.read_text(encoding="utf-8")
            for code, name in mapping.items():
                content = content.replace(code, name)
            f.write_text(content, encoding="utf-8")
        except Exception:
            pass
    _refresh_result(job)
    r = job["result"]
    return jsonify({"ok": True, "summary_md": r.get("summary_md"),
                    "primary": r["primary"], "secondary": r["secondary"]})


@app.route("/download/<path:rel_path>")
def download(rel_path: str):
    full = (OUTPUT_DIR / rel_path).resolve()
    # Path traversal guard
    if not str(full).startswith(str(OUTPUT_DIR.resolve())):
        return jsonify({"error": "Forbidden"}), 403
    if not full.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(str(full), as_attachment=True)


# ── Archive: browse past meetings ────────────────────────────────────────────

@app.route("/meetings")
def list_meetings():
    """Return all meetings found in OUTPUT_DIR, grouped by category."""
    items = []
    if OUTPUT_DIR.exists():
        for meta_path in OUTPUT_DIR.glob("*/*_meta.json"):
            if "_obsidian" in meta_path.parts or "intermediate" in meta_path.parts:
                continue
            meta = _read_meta(meta_path)
            if not meta:
                continue
            category = meta.get("category") or meta_path.parent.name
            stem     = meta.get("stem")     or meta_path.name.replace("_meta.json", "")
            summary  = meta_path.parent / f"{stem}_summary.md"
            if not summary.exists():
                continue
            items.append({
                "title":     meta.get("title", stem),
                "date":      meta.get("date", ""),
                "category":  category,
                "attendees": meta.get("attendees", []),
                "stem":      stem,
                "summary_path": str(summary.relative_to(OUTPUT_DIR)),
                "mtime":     summary.stat().st_mtime,
            })
    # Sort newest first
    items.sort(key=lambda x: (x["date"], x["mtime"]), reverse=True)
    # Group by category
    by_cat: dict[str, list] = {}
    for it in items:
        by_cat.setdefault(it["category"], []).append(it)
    return jsonify({"meetings": items, "by_category": by_cat})


@app.route("/meeting")
def get_meeting():
    """Return summary content + file list for a past meeting.

    Query: ?category=...&stem=...
    """
    category = (request.args.get("category") or "").strip()
    stem     = (request.args.get("stem") or "").strip()
    if not category or not stem:
        return jsonify({"error": "category and stem required"}), 400

    out_dir = (OUTPUT_DIR / category).resolve()
    if not str(out_dir).startswith(str(OUTPUT_DIR.resolve())) or not out_dir.is_dir():
        return jsonify({"error": "Not found"}), 404

    summary_path = out_dir / f"{stem}_summary.md"
    if not summary_path.exists():
        return jsonify({"error": "Summary not found"}), 404

    _PRIMARY = {f"{stem}_summary.md", f"{stem}_speakers.txt",
                f"{stem}_anonymized_speakers.txt"}
    primary, secondary = [], []
    for f in sorted(out_dir.glob(f"{stem}*")):
        if not f.is_file() or "intermediate" in f.parts:
            continue
        entry = {"name": f.name,
                 "path": str(f.relative_to(OUTPUT_DIR)),
                 "size": f.stat().st_size}
        (primary if f.name in _PRIMARY else secondary).append(entry)

    meta = _read_meta(out_dir / f"{stem}_meta.json") or {}
    return jsonify({
        "summary_md": summary_path.read_text(encoding="utf-8"),
        "primary":    primary,
        "secondary":  secondary,
        "meta":       meta,
        "category":   category,
        "stem":       stem,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    INPUT_DIR.mkdir(exist_ok=True)
    app.run(host="0.0.0.0", port=8080, threaded=True)
