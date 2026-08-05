#!/usr/bin/env python3
"""
trello_client.py — одностороннее создание карточек в Trello (ADR-001).

Обратно ничего не читаем в рабочем цикле; статусы живут на доске.
Только stdlib (urllib) — никаких новых зависимостей в образе.

Настройка (.env):
  TRELLO_KEY      — API key  (https://trello.com/power-ups/admin → New → API key)
  TRELLO_TOKEN    — token    (ссылка «Token» рядом с ключом)
  TRELLO_LIST_ID  — id списка «К работе» (открыть доску с .json в URL или
                    GET /1/boards/{id}/lists)
  TRELLO_BOARD_ID — опционально: включает подбор меток и участников по имени
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.trello.com/1"


class TrelloError(Exception):
    pass


def _creds() -> tuple[str, str]:
    return (os.environ.get("TRELLO_KEY", "").strip(),
            os.environ.get("TRELLO_TOKEN", "").strip())


def list_id() -> str:
    return os.environ.get("TRELLO_LIST_ID", "").strip()


def board_id() -> str:
    return os.environ.get("TRELLO_BOARD_ID", "").strip()


def configured() -> bool:
    key, token = _creds()
    return bool(key and token and list_id())


def _call(method: str, path: str, params: dict | None = None) -> object:
    key, token = _creds()
    if not key or not token:
        raise TrelloError("TRELLO_KEY/TRELLO_TOKEN не заданы")
    qs = dict(params or {})
    qs["key"], qs["token"] = key, token
    url = f"{API}{path}"
    data = None
    if method == "GET":
        url += "?" + urllib.parse.urlencode(qs)
    else:
        data = urllib.parse.urlencode(qs).encode()
    req = urllib.request.Request(url, data=data, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        raise TrelloError(f"Trello HTTP {e.code}: {body}") from e
    except Exception as e:
        raise TrelloError(f"Trello: {e}") from e


# Кэш меток/участников доски на процесс — доска меняется редко,
# а создание 10 карточек не должно делать 20 лишних запросов.
_cache: dict[str, list] = {}


def _board_labels() -> list[dict]:
    b = board_id()
    if not b:
        return []
    if "labels" not in _cache:
        try:
            _cache["labels"] = _call("GET", f"/boards/{b}/labels") or []
        except TrelloError:
            _cache["labels"] = []
    return _cache["labels"]


def _board_members() -> list[dict]:
    b = board_id()
    if not b:
        return []
    if "members" not in _cache:
        try:
            _cache["members"] = _call("GET", f"/boards/{b}/members") or []
        except TrelloError:
            _cache["members"] = []
    return _cache["members"]


def _match_label(name: str) -> str | None:
    n = (name or "").strip().lower()
    if not n:
        return None
    for lb in _board_labels():
        if (lb.get("name") or "").strip().lower() == n:
            return lb.get("id")
    return None


def _match_member(owner: str) -> str | None:
    n = (owner or "").strip().lower()
    if not n or n.startswith(("speaker_", "person_")):
        return None
    for m in _board_members():
        full = (m.get("fullName") or "").lower()
        user = (m.get("username") or "").lower()
        if n == full or n == user or (full and n in full.split()):
            return m.get("id")
    return None


def create_card(*, name: str, desc: str, due: str | None = None,
                label_name: str | None = None,
                owner: str | None = None) -> dict:
    """Создать карточку в списке TRELLO_LIST_ID. Возвращает {id, url, ...}.

    Метка и участник — best effort: не нашлись по имени → карточка
    создаётся без них (владелец всё равно записан в описании).
    """
    if not configured():
        raise TrelloError("Trello не настроен: TRELLO_KEY/TOKEN/LIST_ID в .env")
    params: dict = {
        "idList": list_id(),
        "name":   (name or "").strip()[:512] or "Без названия",
        "desc":   (desc or "")[:4000],
        "pos":    "top",
    }
    if due:
        params["due"] = due
    lid = _match_label(label_name or "")
    if lid:
        params["idLabels"] = lid
    mid = _match_member(owner or "")
    if mid:
        params["idMembers"] = mid
    card = _call("POST", "/cards", params)
    if not isinstance(card, dict) or not card.get("id"):
        raise TrelloError(f"Неожиданный ответ Trello: {str(card)[:200]}")
    return card


def get_card(card_id: str) -> dict | None:
    """Разовое чтение карточки для сверки «первый срок против текущего».

    Это НЕ синхронизация (ADR-001): вызывается только кнопкой на Пульсе.
    Удалённая/недоступная карточка → None.
    """
    try:
        card = _call("GET", f"/cards/{card_id}",
                     {"fields": "name,due,dueComplete,closed,shortUrl"})
        return card if isinstance(card, dict) else None
    except TrelloError:
        return None
