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
VAULT_DIR    = Path("/vault")                 # host Obsidian Vault (mounted)
SETTINGS_FILE   = OUTPUT_DIR / "_settings.json"    # UI-editable runtime settings
CATEGORIES_FILE = OUTPUT_DIR / "_categories.json"  # user-defined cat/subcat tree


# ── Runtime settings (UI-editable, persist on disk) ──────────────────────────

# LLM provider presets. Each profile = (label, base_url, default_model, needs_user_api_key).
# `base_url` can be None for "custom" — the user enters their own URL.
LLM_PROFILES: dict[str, dict] = {
    "openai":   {"label": "OpenAI (online)",       "base_url": "https://api.openai.com/v1",   "default_model": "gpt-4o-mini",   "needs_key": True},
    "deepseek": {"label": "DeepSeek (online)",      "base_url": "https://api.deepseek.com/v1", "default_model": "deepseek-chat", "needs_key": True},
    "lmstudio": {"label": "LM Studio (локально)", "base_url": "http://localhost:1234/v1",    "default_model": "qwen3-32b",     "needs_key": False},
    "ollama":   {"label": "Ollama (локально)",    "base_url": "http://localhost:11434/v1",   "default_model": "qwen3:8b",      "needs_key": False},
    "custom":   {"label": "Свой endpoint",         "base_url": None,                            "default_model": "",              "needs_key": True},
}
DEFAULT_LLM_PROFILE = "openai"

# Allowed summary verbosity modes (must match scripts/summarize.py).
SUMMARY_MODES = ("full", "compact", "minimal")
DEFAULT_SUMMARY_MODE = "compact"

# Known settings keys. Values may be str or (for *_MODELS / *_KEYS) dict.
_SETTINGS_KEYS = (
    "LLM_PROFILE",            # active profile id (key of LLM_PROFILES)
    "LLM_PROFILE_MODELS",     # {profile_id: model_name}
    "LLM_PROFILE_KEYS",       # {profile_id: api_key} for online profiles
    "LLM_CUSTOM_BASE_URL",    # only used when profile == "custom"
    "LLM_CUSTOM_API_KEY",     # only used when profile == "custom"
    "LLM_MODEL",              # legacy single-model setting (kept for compat)
    "SUMMARY_MODE",           # full | compact | minimal
    "OBSIDIAN_SUBFOLDER",
)

# Settings keys whose value is a {profile_id: str} dict.
_DICT_SETTINGS_KEYS = ("LLM_PROFILE_MODELS", "LLM_PROFILE_KEYS")


def _load_settings() -> dict:
    """Read settings overrides written by the UI. Missing file → empty dict."""
    try:
        if SETTINGS_FILE.exists():
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            out: dict = {}
            for k in _SETTINGS_KEYS:
                if k not in data:
                    continue
                v = data[k]
                if k in _DICT_SETTINGS_KEYS:
                    if isinstance(v, dict):
                        out[k] = {pk: str(pv) for pk, pv in v.items()
                                  if isinstance(pk, str) and pv}
                elif v not in (None, ""):
                    out[k] = v
            return out
    except Exception:
        pass
    return {}


def _save_settings(data: dict) -> dict:
    current = _load_settings()
    for k in _SETTINGS_KEYS:
        if k not in data:
            continue
        v = data[k]
        if k in _DICT_SETTINGS_KEYS:
            if isinstance(v, dict):
                existing = current.get(k, {}) if isinstance(current.get(k), dict) else {}
                for pk, pv in v.items():
                    if isinstance(pk, str) and isinstance(pv, str) and pv.strip():
                        existing[pk] = pv.strip()
                current[k] = existing
            continue
        if isinstance(v, str):
            v = v.strip()
        if v:
            current[k] = v
        else:
            current.pop(k, None)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(
        json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return current


def _active_llm_profile(settings: dict | None = None) -> str:
    s = settings if settings is not None else _load_settings()
    pid = s.get("LLM_PROFILE") or os.environ.get("LLM_PROFILE") or DEFAULT_LLM_PROFILE
    return pid if pid in LLM_PROFILES else DEFAULT_LLM_PROFILE


def _active_summary_mode(settings: dict | None = None) -> str:
    s = settings if settings is not None else _load_settings()
    mode = (s.get("SUMMARY_MODE") or os.environ.get("SUMMARY_MODE")
            or DEFAULT_SUMMARY_MODE).strip().lower()
    return mode if mode in SUMMARY_MODES else DEFAULT_SUMMARY_MODE


def _profile_has_key(profile_id: str, settings: dict) -> bool:
    """Whether an API key is available for a profile (stored or via env)."""
    if profile_id == "custom":
        return bool(settings.get("LLM_CUSTOM_API_KEY") or os.environ.get("LLM_API_KEY"))
    prof = LLM_PROFILES.get(profile_id)
    if not prof or not prof["needs_key"]:
        return True  # no key required
    keys = settings.get("LLM_PROFILE_KEYS") or {}
    return bool(keys.get(profile_id) or os.environ.get("LLM_API_KEY"))


def _resolve_llm_for_profile(profile_id: str, settings: dict) -> dict:
    """Compute effective {base_url, api_key, model} for a given profile."""
    prof = LLM_PROFILES.get(profile_id, LLM_PROFILES[DEFAULT_LLM_PROFILE])
    profile_models = settings.get("LLM_PROFILE_MODELS") or {}
    profile_keys   = settings.get("LLM_PROFILE_KEYS") or {}
    # Model: per-profile stored → legacy LLM_MODEL (only for active profile) → default
    model = profile_models.get(profile_id) or ""
    if not model and profile_id == _active_llm_profile(settings):
        model = settings.get("LLM_MODEL") or os.environ.get("LLM_MODEL") or ""
    if not model:
        model = prof["default_model"]

    if profile_id == "custom":
        base_url = (settings.get("LLM_CUSTOM_BASE_URL")
                    or os.environ.get("LLM_BASE_URL") or "")
        api_key  = (settings.get("LLM_CUSTOM_API_KEY")
                    or os.environ.get("LLM_API_KEY") or "not-required")
    else:
        base_url = prof["base_url"] or ""
        if prof["needs_key"]:
            api_key = (profile_keys.get(profile_id)
                       or os.environ.get("LLM_API_KEY") or "")
        else:
            api_key = "not-required"
    return {"base_url": base_url, "api_key": api_key, "model": model}


def _effective_env() -> dict:
    """Return os.environ merged with persisted UI settings, applying active LLM profile."""
    env = {**os.environ}
    settings = _load_settings()
    profile_id = _active_llm_profile(settings)
    resolved = _resolve_llm_for_profile(profile_id, settings)
    if resolved["base_url"]:
        env["LLM_BASE_URL"] = resolved["base_url"]
    if resolved["api_key"]:
        env["LLM_API_KEY"]  = resolved["api_key"]
    if resolved["model"]:
        env["LLM_MODEL"]    = resolved["model"]
    env["LLM_PROFILE"]  = profile_id
    env["SUMMARY_MODE"] = _active_summary_mode(settings)
    sub = settings.get("OBSIDIAN_SUBFOLDER")
    if sub:
        env["OBSIDIAN_SUBFOLDER"] = sub
    return env


def _vault_subfolder() -> str:
    """Resolve subfolder name inside /vault — env or settings override."""
    s = _load_settings().get("OBSIDIAN_SUBFOLDER")
    if s:
        return s
    return (os.environ.get("OBSIDIAN_SUBFOLDER") or "Meetings").strip()


def _vault_available() -> bool:
    """Check if the Obsidian Vault is actually mounted and writable."""
    try:
        return VAULT_DIR.is_dir()
    except Exception:
        return False


# ── Categories store (user-defined Category → Subcategory tree) ──────────────

_DEFAULT_CATEGORIES = {
    "Общее": ["Без подкатегории"],
}


def _load_categories() -> dict:
    """Return {category_name: [subcategory, …], …}. Initialize file on first run."""
    try:
        if CATEGORIES_FILE.exists():
            data = json.loads(CATEGORIES_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                # Normalise: every value must be a list of strings
                out: dict[str, list[str]] = {}
                for k, v in data.items():
                    if isinstance(k, str) and k.strip():
                        subs = v if isinstance(v, list) else []
                        out[k.strip()] = [s.strip() for s in subs
                                          if isinstance(s, str) and s.strip()]
                return out
    except Exception:
        pass
    return dict(_DEFAULT_CATEGORIES)


def _save_categories(data: dict) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CATEGORIES_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return data


_CAT_NAME_RE = re.compile(r"[<>:\"/\\|?*\x00-\x1f]")


def _valid_cat_name(name: str) -> bool:
    if not name or not isinstance(name, str):
        return False
    name = name.strip()
    if not name or len(name) > 80:
        return False
    if name in (".", ".."):
        return False
    return _CAT_NAME_RE.search(name) is None


_DASHBOARD_TEMPLATE = """\
---
title: "Meetings Dashboard"
tags: [meetings, dashboard]
---

# 📊 Meetings Dashboard

> Эта страница автоматически создана плагином Meeting Transcriber.
> Используется плагином **Dataview** для агрегации заметок.

## 🔥 Свежие планёрки

```dataview
TABLE WITHOUT ID
  file.link AS "Планёрка",
  date AS "Дата",
  category AS "Категория",
  length(attendees) AS "Участники"
FROM "{folder}"
WHERE type != "dashboard"
SORT date DESC
LIMIT 20
```

## ✅ Открытые задачи

```dataview
TASK FROM "{folder}"
WHERE !completed
SORT due ASC
```

## 🗂 По категориям

```dataview
TABLE length(rows) AS "Кол-во"
FROM "{folder}"
WHERE type != "dashboard"
GROUP BY category
SORT length(rows) DESC
```
"""


def _ensure_dashboard(folder: Path, subfolder: str) -> None:
    """Create Dashboard.md in the vault subfolder once (don't overwrite)."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        dash = folder / "Dashboard.md"
        if not dash.exists():
            dash.write_text(
                _DASHBOARD_TEMPLATE.format(folder=subfolder),
                encoding="utf-8",
            )
    except Exception:
        pass

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


def _create_obsidian_note(out_dir: Path, stem: str,
                          category: str, subcategory: str) -> Path | None:
    """Create a nicely-named Obsidian-friendly copy of the summary.

    Reads {stem}_summary.md (already includes YAML frontmatter) and
    {stem}_meta.json (for title/date/attendees) and writes it under
    OUTPUT_DIR/_obsidian/{Category}/{Subcategory}/{YYYY-MM-DD} — {Title}.md
    plus a sibling copy of the (non-anonymized) speakers transcript.
    Also mirrors both files into the mounted Obsidian Vault when available.
    """
    meta_path     = out_dir / f"{stem}_meta.json"
    summary_path  = out_dir / f"{stem}_summary.md"
    speakers_path = out_dir / f"{stem}_speakers.txt"
    if not (meta_path.exists() and summary_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    title     = meta.get("title", "Совещание")
    date      = meta.get("date", "")
    attendees = [a for a in meta.get("attendees", []) if a and not a.startswith("SPEAKER_")]

    content = summary_path.read_text(encoding="utf-8")

    # Wrap attendee names in [[wiki-links]] inside the body
    if attendees:
        body_start = content.find("\n\n", content.find("---\n", 4) + 4) if content.startswith("---") else 0
        head, body = content[:body_start], content[body_start:]
        for name in sorted(attendees, key=len, reverse=True):
            pattern = re.compile(rf"(?<!\[)\b{re.escape(name)}\b(?!\]\])")
            body = pattern.sub(f"[[{name}]]", body)
        content = head + body

    cat_slug = _slugify_filename(category, 40) or "Общее"
    sub_slug = _slugify_filename(subcategory, 40) or "Без подкатегории"

    base_name = f"{date} — {_slugify_filename(title)}" if date else _slugify_filename(title)
    summary_fname    = f"{base_name}.md"
    transcript_fname = f"{base_name} — транскрипт.md"

    transcript_md = None
    if speakers_path.exists():
        try:
            raw = speakers_path.read_text(encoding="utf-8")
            transcript_md = (
                f"---\n"
                f"type: meeting-transcript\n"
                f"meeting: \"[[{base_name}]]\"\n"
                f"date: {date}\n"
                f"category: {category}\n"
                f"subcategory: {subcategory}\n"
                f"---\n\n"
                f"# Транскрипт — {title}\n\n"
                f"> Связанная заметка: [[{base_name}]]\n\n"
                f"```\n{raw}\n```\n"
            )
        except Exception:
            transcript_md = None

    # ── Local mirror inside /output/_obsidian/ ───────────────────────────────
    folder = OBSIDIAN_DIR / cat_slug / sub_slug
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / summary_fname
    dest.write_text(content, encoding="utf-8")
    if transcript_md:
        (folder / transcript_fname).write_text(transcript_md, encoding="utf-8")

    # ── Also write to mounted Obsidian Vault if available ────────────────────
    if _vault_available():
        try:
            sub = _vault_subfolder()
            vault_root = VAULT_DIR / sub
            _ensure_dashboard(vault_root, sub)
            cat_folder = vault_root / cat_slug / sub_slug
            cat_folder.mkdir(parents=True, exist_ok=True)
            (cat_folder / summary_fname).write_text(content, encoding="utf-8")
            if transcript_md:
                (cat_folder / transcript_fname).write_text(transcript_md, encoding="utf-8")
        except Exception:
            pass

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


def _pipeline(job_id: str, audio_path: Path, category: str, subcategory: str,
              language: str, llm_enabled: bool) -> None:
    """Main pipeline — runs in a background daemon thread."""
    job = _jobs[job_id]
    stem    = audio_path.stem
    cat_slug = _slugify_filename(category, 40) or "Общее"
    sub_slug = _slugify_filename(subcategory, 40) or "Без подкатегории"
    out_subdir = f"{cat_slug}/{sub_slug}"
    out_dir = OUTPUT_DIR / cat_slug / sub_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    job["status"] = "running"
    job["out_dir"] = out_dir
    job["stem"]    = stem

    try:
        # ── Build environment ────────────────────────────────────────────────────
        env = _effective_env()
        env["OUTPUT_SUBDIR"] = out_subdir
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
            "summary_md": summary_md,
            "category": category, "subcategory": subcategory,
        }
        # ── Create Obsidian-friendly copy (if meta available) ────────────────
        try:
            obs = _create_obsidian_note(out_dir, stem, category, subcategory)
            if obs:
                _log(job, f"📒 Obsidian-заметка: {obs.relative_to(OUTPUT_DIR)}", "ok")
        except Exception as e:
            _log(job, f"Не удалось создать Obsidian-заметку: {e}", "warn")

        job["status"] = "done"
        _log(job, f"✅ Готово! Файлы сохранены в output/{out_subdir}/", "ok")

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


@app.route("/settings")
def settings_page():
    return render_template("settings.html")


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    """Read/update UI-editable runtime settings (LLM profile/model, Obsidian subfolder)."""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        _save_settings(data)
    persisted = _load_settings()
    active = _active_llm_profile(persisted)
    resolved_per_profile = {
        pid: _resolve_llm_for_profile(pid, persisted) for pid in LLM_PROFILES
    }
    # Redact secrets from the settings echo — never send raw API keys to the UI.
    safe_settings = {k: v for k, v in persisted.items()
                     if k not in ("LLM_PROFILE_KEYS", "LLM_CUSTOM_API_KEY")}
    return jsonify({
        "settings": safe_settings,
        "active_profile": active,
        "summary_mode": _active_summary_mode(persisted),
        "summary_modes": list(SUMMARY_MODES),
        "profiles": {
            pid: {
                "label":         prof["label"],
                "base_url":      resolved_per_profile[pid]["base_url"],
                "default_model": prof["default_model"],
                "needs_key":     prof["needs_key"],
                "has_key":       _profile_has_key(pid, persisted),
                "model":         resolved_per_profile[pid]["model"],
                "editable_url":  pid == "custom",
            } for pid, prof in LLM_PROFILES.items()
        },
        "effective": {
            "LLM_PROFILE":        active,
            "LLM_BASE_URL":       resolved_per_profile[active]["base_url"],
            "LLM_MODEL":          resolved_per_profile[active]["model"],
            "SUMMARY_MODE":       _active_summary_mode(persisted),
            "OBSIDIAN_SUBFOLDER": persisted.get("OBSIDIAN_SUBFOLDER") or os.environ.get("OBSIDIAN_SUBFOLDER", "Meetings"),
        },
        "llm_base_url":    resolved_per_profile[active]["base_url"],
        "vault_mounted":   _vault_available(),
        "vault_subfolder": _vault_subfolder(),
    })


@app.route("/api/llm-models")
def api_llm_models():
    """Proxy to {LLM_BASE_URL}/models — returns list of available model ids.

    Optional query params let the UI probe a profile that isn't saved yet:
      ?profile=<id>       — resolve base_url/key for that profile
      ?base_url=<url>     — override base url (for custom, unsaved)
      ?api_key=<key>      — override api key (for unsaved key entry)
    """
    import urllib.request
    import urllib.error

    settings = _load_settings()
    profile_id = request.args.get("profile")
    if profile_id and profile_id in LLM_PROFILES:
        resolved = _resolve_llm_for_profile(profile_id, settings)
        base = (resolved["base_url"] or "").rstrip("/")
        api_key = resolved["api_key"] or "not-required"
    else:
        base = (_effective_env().get("LLM_BASE_URL") or "").rstrip("/")
        api_key = _effective_env().get("LLM_API_KEY") or "not-required"

    # Allow explicit overrides (unsaved custom URL / freshly typed key).
    override_url = (request.args.get("base_url") or "").strip()
    if override_url:
        base = override_url.rstrip("/")
    override_key = (request.args.get("api_key") or "").strip()
    if override_key:
        api_key = override_key

    if not base:
        return jsonify({"models": [], "error": "LLM_BASE_URL not set"}), 200

    # Rewrite localhost → host.docker.internal for in-container calls
    url = base
    for host in ("localhost", "127.0.0.1"):
        if f"://{host}" in url:
            url = url.replace(f"://{host}", "://host.docker.internal")
            break
    url = url + "/models"

    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        items = data.get("data", []) if isinstance(data, dict) else []
        models = sorted({m.get("id") for m in items if m.get("id")})
        return jsonify({"models": models})
    except Exception as exc:
        return jsonify({"models": [], "error": str(exc)}), 200


@app.route("/api/vault-browse")
def api_vault_browse():
    """List directories inside the mounted /vault for the folder picker.

    Query: ?path=relative/subdir   (relative to /vault, never escapes it)
    """
    if not _vault_available():
        return jsonify({"mounted": False, "path": "", "dirs": []})

    rel = (request.args.get("path") or "").strip().strip("/").replace("\\", "/")
    try:
        target = (VAULT_DIR / rel).resolve()
        # Confine inside /vault
        VAULT_DIR_RESOLVED = VAULT_DIR.resolve()
        if target != VAULT_DIR_RESOLVED and VAULT_DIR_RESOLVED not in target.parents:
            target = VAULT_DIR_RESOLVED
            rel = ""
        if not target.is_dir():
            return jsonify({"mounted": True, "path": rel, "dirs": [], "error": "not a directory"})
        dirs = sorted(
            [p.name for p in target.iterdir() if p.is_dir() and not p.name.startswith(".")],
            key=str.lower,
        )
        return jsonify({"mounted": True, "path": rel, "dirs": dirs})
    except Exception as exc:
        return jsonify({"mounted": True, "path": rel, "dirs": [], "error": str(exc)}), 200


@app.route("/api/vault-mkdir", methods=["POST"])
def api_vault_mkdir():
    """Create a new subfolder inside /vault. Body: {path: 'rel/path', name: 'NewFolder'}"""
    if not _vault_available():
        return jsonify({"ok": False, "error": "vault not mounted"}), 400
    data = request.get_json(silent=True) or {}
    rel  = (data.get("path") or "").strip().strip("/").replace("\\", "/")
    name = (data.get("name") or "").strip()
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        return jsonify({"ok": False, "error": "invalid name"}), 400
    try:
        parent = (VAULT_DIR / rel).resolve()
        VAULT_DIR_RESOLVED = VAULT_DIR.resolve()
        if parent != VAULT_DIR_RESOLVED and VAULT_DIR_RESOLVED not in parent.parents:
            return jsonify({"ok": False, "error": "outside vault"}), 400
        (parent / name).mkdir(parents=True, exist_ok=True)
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 200


@app.route("/api/categories", methods=["GET", "POST"])
def api_categories():
    """List or create user-defined categories.

    GET  -> {"categories": {name: [subcategories]}}
    POST -> body {"name": "<Cat>"} adds a top-level category.
    """
    if request.method == "GET":
        return jsonify({"categories": _load_categories()})
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not _valid_cat_name(name):
        return jsonify({"ok": False, "error": "недопустимое имя"}), 400
    cats = _load_categories()
    if name in cats:
        return jsonify({"ok": False, "error": "уже существует"}), 409
    cats[name] = []
    _save_categories(cats)
    return jsonify({"ok": True, "categories": cats})


@app.route("/api/categories/<path:cat>", methods=["DELETE"])
def api_categories_delete(cat: str):
    cat = (cat or "").strip()
    cats = _load_categories()
    if cat not in cats:
        return jsonify({"ok": False, "error": "не найдена"}), 404
    del cats[cat]
    _save_categories(cats)
    return jsonify({"ok": True, "categories": cats})


@app.route("/api/categories/<path:cat>/subcategories", methods=["POST"])
def api_subcategories_add(cat: str):
    cat = (cat or "").strip()
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not _valid_cat_name(name):
        return jsonify({"ok": False, "error": "недопустимое имя"}), 400
    cats = _load_categories()
    if cat not in cats:
        return jsonify({"ok": False, "error": "категория не найдена"}), 404
    if name in cats[cat]:
        return jsonify({"ok": False, "error": "уже существует"}), 409
    cats[cat].append(name)
    _save_categories(cats)
    return jsonify({"ok": True, "categories": cats})


@app.route("/api/categories/<path:cat>/subcategories/<path:sub>", methods=["DELETE"])
def api_subcategories_delete(cat: str, sub: str):
    cat = (cat or "").strip()
    sub = (sub or "").strip()
    cats = _load_categories()
    if cat not in cats or sub not in cats.get(cat, []):
        return jsonify({"ok": False, "error": "не найдена"}), 404
    cats[cat] = [s for s in cats[cat] if s != sub]
    _save_categories(cats)
    return jsonify({"ok": True, "categories": cats})


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

    category    = (request.form.get("category") or "").strip()
    subcategory = (request.form.get("subcategory") or "").strip()
    language    = request.form.get("language", "ru")
    llm_enabled = request.form.get("llm_enabled", "true") == "true"

    if not _valid_cat_name(category) or not _valid_cat_name(subcategory):
        return jsonify({"error": "Выберите категорию и подкатегорию перед запуском"}), 400

    # Auto-register the chosen pair so it persists for next time
    cats = _load_categories()
    if category not in cats:
        cats[category] = []
    if subcategory not in cats[category]:
        cats[category].append(subcategory)
    _save_categories(cats)

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status":      "pending",
        "result":      None,
        "filename":    filename,
        "category":    category,
        "subcategory": subcategory,
        "started_at":  time.time(),
        "finished_at": None,
        "log_buffer":  [],     # list of message dicts (replayed on (re)connect)
        "subscribers": [],     # list of queue.Queue, one per active SSE client
    }

    threading.Thread(
        target=_pipeline,
        args=(job_id, audio_path, category, subcategory, language, llm_enabled),
        daemon=True,
    ).start()

    return jsonify({"job_id": job_id, "filename": filename,
                    "category": category, "subcategory": subcategory})


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
