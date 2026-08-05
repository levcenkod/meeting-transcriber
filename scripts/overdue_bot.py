#!/usr/bin/env python3
"""
overdue_bot.py — ежедневное уведомление о просроченных карточках Trello.

Читает ТОЛЬКО доску Trello и шлёт ТОЛЬКО в Telegram: ни GPU, ни локальных
файлов транскрайбера не нужно — поэтому запускается где угодно (основной
хостинг — GitHub Actions cron, см. .github/workflows/overdue-bot.yml;
локальный запуск и планировщик Windows — fallback).

Правила (ТЗ v0, раздел 5):
  - одно сообщение со всеми просрочками, не по карточке;
  - не чаще одного сообщения в день (страховка через файл состояния,
    в Actions частоту и так задаёт cron);
  - нет просрочек — молчим.

Просрочка: у карточки есть due в прошлом, dueComplete=false и карточка
не в «готовом» списке (TRELLO_DONE_LISTS, по умолчанию Готово/Done).

Env: TRELLO_KEY, TRELLO_TOKEN, TRELLO_BOARD_ID,
     TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
     TRELLO_DONE_LISTS (опц.), BOT_STATE_FILE (опц.), TZ (опц.)

CLI: python overdue_bot.py [--dry-run]
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from trello_client import TrelloError, _call, board_id  # noqa: E402
import telegram_notify  # noqa: E402

_DEFAULT_DONE = "Готово,✅ Готово,Done,✅ Done,Сделано"
_MAX_LINES = 15


def _done_list_ids() -> set[str]:
    names = {n.strip().lower()
             for n in os.environ.get("TRELLO_DONE_LISTS", _DEFAULT_DONE).split(",")
             if n.strip()}
    ids = set()
    for lst in _call("GET", f"/boards/{board_id()}/lists") or []:
        lname = (lst.get("name") or "").strip().lower()
        if lname in names or any(n in lname for n in names):
            ids.add(lst.get("id"))
    return ids


def _fmt_date(iso: str) -> str:
    try:
        return dt.datetime.fromisoformat(iso.replace("Z", "+00:00")) \
                 .astimezone().strftime("%d.%m")
    except ValueError:
        return iso[:10]


def collect_overdue() -> list[dict]:
    if not board_id():
        raise TrelloError("TRELLO_BOARD_ID не задан")
    done_ids = _done_list_ids()
    now = dt.datetime.now(dt.timezone.utc)
    cards = _call(
        "GET", f"/boards/{board_id()}/cards",
        {"fields": "name,due,dueComplete,idList,shortUrl", "filter": "open"},
    ) or []
    out = []
    for c in cards:
        due = c.get("due")
        if not due or c.get("dueComplete") or c.get("idList") in done_ids:
            continue
        try:
            due_dt = dt.datetime.fromisoformat(due.replace("Z", "+00:00"))
        except ValueError:
            continue
        if due_dt < now:
            out.append({
                "name": c.get("name") or "Без названия",
                "due": due,
                "days": (now - due_dt).days,
                "url": c.get("shortUrl") or "",
            })
    out.sort(key=lambda x: -x["days"])
    return out


def build_message(overdue: list[dict]) -> str:
    lines = [f"⏰ Просроченных задач на доске: {len(overdue)}"]
    for c in overdue[:_MAX_LINES]:
        days = f"+{c['days']} дн." if c["days"] > 0 else "сегодня"
        lines.append(f"• {c['name']} — до {_fmt_date(c['due'])} ({days})")
    if len(overdue) > _MAX_LINES:
        lines.append(f"…и ещё {len(overdue) - _MAX_LINES}")
    lines.append("Статусы двигаем в Trello, бот только напоминает.")
    return "\n".join(lines)


def _state_file() -> Path:
    p = os.environ.get("BOT_STATE_FILE")
    return Path(p) if p else Path(__file__).parent / ".overdue_bot_state.json"


def _already_sent_today() -> bool:
    try:
        state = json.loads(_state_file().read_text(encoding="utf-8"))
        return state.get("last_sent") == dt.date.today().isoformat()
    except Exception:
        return False


def _mark_sent() -> None:
    try:
        _state_file().write_text(
            json.dumps({"last_sent": dt.date.today().isoformat()}),
            encoding="utf-8")
    except OSError:
        pass


def main() -> int:
    dry = "--dry-run" in sys.argv
    try:
        overdue = collect_overdue()
    except TrelloError as e:
        print(f"[BOT][ERROR] {e}", file=sys.stderr)
        return 1
    if not overdue:
        print("[BOT] Просрочек нет — молчим.")
        return 0
    msg = build_message(overdue)
    print(msg)
    if dry:
        return 0
    if _already_sent_today():
        print("[BOT] Сегодня уже отправляли — не чаще одного сообщения в день.")
        return 0
    if not telegram_notify.configured():
        print("[BOT][ERROR] TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID не заданы",
              file=sys.stderr)
        return 1
    if telegram_notify.send(msg):
        _mark_sent()
        print("[BOT] Отправлено.")
        return 0
    print("[BOT][ERROR] Telegram не принял сообщение", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
