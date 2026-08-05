#!/usr/bin/env python3
"""
telegram_notify.py — отправка сообщений в Telegram (stdlib only).

Используется дважды:
  - зеркало журнала отправки (страховка первых сроков, ADR-002)
  - бот просрочек (scripts/overdue_bot.py)

Настройка (.env / секреты GitHub Actions):
  TELEGRAM_BOT_TOKEN — токен бота от @BotFather
  TELEGRAM_CHAT_ID   — id чата/канала (бот должен быть участником)
"""

from __future__ import annotations

import json
import os
import urllib.request


def configured() -> bool:
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
                and os.environ.get("TELEGRAM_CHAT_ID", "").strip())


def send(text: str, *, silent: bool = False) -> bool:
    """Отправить сообщение. False при любой ошибке — зеркало не роняет отправку."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat or not text:
        return False
    payload = json.dumps({
        "chat_id": chat,
        "text": text[:4000],
        "disable_notification": silent,
        "disable_web_page_preview": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return bool(json.loads(r.read().decode("utf-8")).get("ok"))
    except Exception:
        return False
