#!/usr/bin/env python3
"""
people.py — реестр людей команды и нормализация имён владельцев.

Владельцы из LLM приходят в падежах и с мусором: «Владу», «Марии»,
«Дизайнеров», «Андрей (подрядчик)», «SPEAKER_00», «PERSON_003», «Ладно».
Нормализация сводит их к канону из реестра (`output/_people.json`);
что не свелось — остаётся как есть, решает человек в приёмке.

Стемминг грубый и рассчитан на маленький список имён команды (5–10 человек),
где коллизии («Влад»/«Влада») маловероятны и видны глазами.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_PEOPLE_FILE = Path("/output/_people.json")

_VOWELS = "аеёиоуыэюя"
_SPEAKER_RE = re.compile(r"^(SPEAKER|PERSON|UNKNOWN)[_ ]?", re.IGNORECASE)
_EMPTY_VALUES = {"", "-", "—", "null", "none", "не назначен", "не назначено",
                 "не указан", "не указано", "unknown", "n/a", "tbd"}


def load_people(path: Path | None = None) -> list[str]:
    p = path or DEFAULT_PEOPLE_FILE
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            seen, out = set(), []
            for s in data:
                if isinstance(s, str) and s.strip() and s.strip() not in seen:
                    seen.add(s.strip())
                    out.append(s.strip())
            return out
    except Exception:
        pass
    return []


def save_people(people: list[str], path: Path | None = None) -> list[str]:
    p = path or DEFAULT_PEOPLE_FILE
    clean, seen = [], set()
    for s in people:
        if isinstance(s, str) and s.strip() and s.strip().lower() not in seen:
            seen.add(s.strip().lower())
            clean.append(s.strip())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    return clean


def is_speaker_label(name: str | None) -> bool:
    """SPEAKER_00 / PERSON_003 / UNKNOWN_SPEAKER — метка голоса, не имя."""
    return bool(name) and bool(_SPEAKER_RE.match(name.strip()))


def _stem_ru(word: str) -> str:
    """Грубый стем для сравнения падежей: срезать конечные гласные/й/ь."""
    w = word.lower().replace("ё", "е")
    while len(w) > 2 and w[-1] in _VOWELS + "йь":
        w = w[:-1]
    return w


def _match_one(token: str, people: list[str]) -> str | None:
    tl = token.lower().replace("ё", "е")
    for person in people:
        if person.lower().replace("ё", "е") == tl:
            return person
    ts = _stem_ru(token)
    for person in people:
        ps = _stem_ru(person)
        if ts == ps:
            return person
        # «Дизайнеров» → «Дизайнер»: один стем — префикс другого, разница ≤ 3
        longer, shorter = (ts, ps) if len(ts) >= len(ps) else (ps, ts)
        if len(shorter) >= 3 and longer.startswith(shorter) and len(longer) - len(shorter) <= 3:
            return person
    return None


def normalize_owner(raw: str | None, people: list[str]) -> tuple[str | None, bool]:
    """Свести владельца к канону. Возвращает (имя | метка | None, matched).

    - None/«не назначен»/пустое → (None, False)
    - SPEAKER_XX/PERSON_XX      → (метка как есть, False) — маппится в приёмке
    - «Владу», «Марии (уточнить)» → (канон из реестра, True), если свёлся
    - иначе                      → (очищенная строка, False)
    """
    if raw is None:
        return None, False
    s = re.sub(r"\s+", " ", str(raw)).strip()
    if s.lower() in _EMPTY_VALUES:
        return None, False
    if is_speaker_label(s):
        return s, False
    # отрезать пояснения в скобках и после запятой: «Андрей (подрядчик)» → «Андрей»
    base = re.split(r"[(,/]| или | и ", s)[0].strip() or s
    hit = _match_one(base, people)
    if not hit and " " in base:
        hit = _match_one(base.split()[0], people)
    if hit:
        return hit, True
    return s, False
