#!/usr/bin/env python3
"""
tg_ingest.py — телеграм-воронка: кинул боту войс/аудио → файл упал в /inbox,
дальше его подхватывает обычная воронка транскрайбера.

Безопасность в концепции проекта (ADR-002, tailscale-переезд):
  - только исходящие запросы к api.telegram.org (long polling) — на сервере
    не открывается ни один порт;
  - файлы принимаются ТОЛЬКО от разрешённых отправителей:
    TELEGRAM_ALLOWED_IDS (id через запятую), а если он пуст — от отправителя
    или чата, совпадающего с TELEGRAM_CHAT_ID (рабочий чат уведомлений);
  - остальным бот отвечает отказом и показывает их id — так удобно
    добавлять нового человека в список.

Ограничение Bot API: боту отдаются файлы до ~20 МБ. Час войса ≈ 8–14 МБ —
проходит; большие записи — через страницу /upload или папку inbox.

Имя сохранённого файла содержит дату сообщения (tg_2026-08-14_1030_voice.ogg),
поэтому дата планёрки определяется парсером meeting_date по имени файла.
Подпись к файлу вида «Категория/Подкатегория» кладёт запись в подпапку
inbox — воронка возьмёт категорию оттуда.

Env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID и/или TELEGRAM_ALLOWED_IDS,
     INBOX_DIR (default /inbox), TZ.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.telegram.org"
MAX_BOT_FILE = 19_900_000          # лимит Bot API ~20 МБ, чуть с запасом
POLL_TIMEOUT = 50

INBOX_DIR = Path(os.environ.get("INBOX_DIR", "/inbox"))

_FS_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _token() -> str:
    return os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()


def _allowed_ids() -> set[str]:
    ids = {s.strip() for s in os.environ.get("TELEGRAM_ALLOWED_IDS", "").split(",")
           if s.strip()}
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if chat:
        ids.add(chat)
    return ids


def _call(method: str, params: dict | None = None, timeout: int = 60) -> dict:
    data = json.dumps(params or {}).encode("utf-8")
    req = urllib.request.Request(
        f"{API}/bot{_token()}/{method}", data=data,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _reply(chat_id, text: str, reply_to=None) -> None:
    try:
        params = {"chat_id": chat_id, "text": text[:4000],
                  "disable_web_page_preview": True}
        if reply_to:
            params["reply_to_message_id"] = reply_to
        _call("sendMessage", params, timeout=15)
    except Exception as e:
        print(f"[TG] reply failed: {e}", flush=True)


def _pick_attachment(msg: dict) -> tuple[dict, str, str] | None:
    """(file-объект, имя_для_файла, метка_типа) или None."""
    if "voice" in msg:
        return msg["voice"], "voice.ogg", "войс"
    if "audio" in msg:
        a = msg["audio"]
        return a, a.get("file_name") or "audio.mp3", "аудио"
    if "video_note" in msg:
        return msg["video_note"], "video_note.mp4", "кружок"
    if "video" in msg:
        v = msg["video"]
        return v, v.get("file_name") or "video.mp4", "видео"
    if "document" in msg:
        d = msg["document"]
        name = d.get("file_name") or "file.bin"
        ext = Path(name).suffix.lower()
        if ext in {".mp3", ".wav", ".m4a", ".ogg", ".mp4", ".mkv",
                   ".flac", ".aac", ".webm", ".opus"}:
            return d, name, "файл"
    return None


def _category_subdir(caption: str) -> Path:
    """Подпись «Категория/Подкатегория» → подпапка внутри inbox."""
    rel = Path()
    for part in (caption or "").strip().split("/")[:2]:
        part = _FS_UNSAFE.sub("-", part).strip().strip(".")
        if part and part not in (".", ".."):
            rel = rel / part
    return rel


def _save_name(msg: dict, orig_name: str) -> str:
    """tg_2026-08-14_1030_voice.ogg — дата сообщения попадает в имя файла,
    и парсер даты планёрки берёт её оттуда."""
    stamp = dt.datetime.fromtimestamp(int(msg.get("date", time.time())))
    base = _FS_UNSAFE.sub("-", Path(orig_name).stem)[:60] or "recording"
    ext = Path(orig_name).suffix.lower() or ".bin"
    return f"tg_{stamp:%Y-%m-%d_%H%M}_{base}{ext}"


def _download(file_id: str, dest: Path) -> None:
    info = _call("getFile", {"file_id": file_id}, timeout=30)
    fpath = (info.get("result") or {}).get("file_path")
    if not fpath:
        raise RuntimeError(f"getFile не вернул путь: {str(info)[:200]}")
    url = f"{API}/file/bot{_token()}/{fpath}"
    tmp = dest.with_suffix(dest.suffix + ".part")   # .part скипается воронкой
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)


def _handle_message(msg: dict) -> None:
    chat_id = (msg.get("chat") or {}).get("id")
    from_id = (msg.get("from") or {}).get("id")
    mid = msg.get("message_id")
    allowed = _allowed_ids()
    # Лог каждого входящего: иначе «кинул и тишина» не отладить.
    kinds = [k for k in ("voice", "audio", "video", "video_note", "document",
                         "text", "photo", "sticker") if k in msg]
    print(f"[TG] update: from={from_id} chat={chat_id} "
          f"type={','.join(kinds) or '?'} "
          f"allowed={'yes' if {str(from_id), str(chat_id)} & allowed else 'NO'}",
          flush=True)

    if not ({str(from_id), str(chat_id)} & allowed):
        _reply(chat_id,
               f"⛔ Доступ не настроен. Ваш id: {from_id} (чат: {chat_id}).\n"
               f"Добавьте его в TELEGRAM_ALLOWED_IDS в .env транскрайбера "
               f"и перезапустите воронку.", mid)
        return

    att = _pick_attachment(msg)
    if att is None:
        if any(k in msg for k in ("text", "sticker", "photo")):
            _reply(chat_id,
                   "Я принимаю записи планёрок: войс, аудио или видео файлом.\n"
                   "Подпись к файлу «Категория/Подкатегория» — положит запись "
                   "в нужное направление.", mid)
        return

    fobj, orig_name, kind = att
    size = int(fobj.get("file_size") or 0)
    if size > MAX_BOT_FILE:
        _reply(chat_id,
               f"⚠️ Файл {size / 1e6:.0f} МБ — больше лимита бота (20 МБ).\n"
               f"Залейте его через страницу /upload — она в тейлнете.", mid)
        return

    rel = _category_subdir(msg.get("caption") or "")
    dest = INBOX_DIR / rel / _save_name(msg, orig_name)
    try:
        _download(fobj["file_id"], dest)
    except Exception as e:
        print(f"[TG] download failed: {e}", flush=True)
        _reply(chat_id, f"✗ Не смог скачать файл: {e}", mid)
        return

    where = f" → {rel.as_posix()}" if rel.parts else ""
    print(f"[TG] Принят {kind}: {dest.name}{where}", flush=True)
    _reply(chat_id,
           f"🎙 Принял {kind} ({size / 1e6:.1f} МБ){where}.\n"
           f"Обрабатываю в фоне — как закончу, напишу сюда. "
           f"Задачи появятся в приёмке.", mid)


def main() -> int:
    if not _token():
        print("[TG] TELEGRAM_BOT_TOKEN не задан — воронка спит.", flush=True)
        while True:                      # не выходим: иначе restart-петля compose
            time.sleep(3600)
    if not _allowed_ids():
        print("[TG] ВНИМАНИЕ: TELEGRAM_ALLOWED_IDS/CHAT_ID пусты — "
              "бот будет всем отказывать и показывать их id.", flush=True)
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[TG] Воронка активна → {INBOX_DIR}", flush=True)

    offset = 0
    while True:
        try:
            upd = _call("getUpdates",
                        {"offset": offset, "timeout": POLL_TIMEOUT,
                         "allowed_updates": ["message"]},
                        timeout=POLL_TIMEOUT + 15)
            for u in upd.get("result", []):
                offset = max(offset, int(u["update_id"]) + 1)
                if "message" in u:
                    try:
                        _handle_message(u["message"])
                    except Exception as e:
                        print(f"[TG] handle error: {e}", flush=True)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"[TG] poll error: {e} — жду 15 с", flush=True)
            time.sleep(15)
        except Exception as e:
            print(f"[TG] unexpected: {e} — жду 30 с", flush=True)
            time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
