#!/usr/bin/env python3
"""
meeting_date.py — определение настоящей даты планёрки.

Дата встречи ≠ дата обработки файла. Порядок источников (resolve_meeting_date):
  1. Явно заданная дата (поле формы → env MEETING_DATE).
  2. Дата из имени файла: "bandicam 2026-06-22 11-52-27", "22.06.2026", "20260622".
  3. mtime исходного файла (если передан).
  4. None — вызывающий решает сам (у summarize.py это сегодня, с warning'ом).

Один раз записанная в meta.json дата при переобработке не пересчитывается —
это ответственность вызывающего (см. summarize.py: existing_meta_date).
"""

from __future__ import annotations

import datetime as _dt
import re

# Разумное окно для дат из имён файлов: старее 2020 и будущее дальше чем
# на 2 дня — почти наверняка не дата планёрки, а мусор в имени.
_MIN_YEAR = 2020
_MAX_FUTURE_DAYS = 2

# Порядок паттернов важен: сначала более однозначные.
_PATTERNS = (
    # 2026-06-22 / 2026_06_22 / 2026.06.22
    re.compile(r"(?<!\d)(20\d{2})[-_.](\d{1,2})[-_.](\d{1,2})(?!\d)"),
    # 22.06.2026 / 22-06-2026 / 22_06_2026
    re.compile(r"(?<!\d)(\d{1,2})[-_.](\d{1,2})[-_.](20\d{2})(?!\d)"),
    # 20260622 (компактно, ровно 8 цифр, не часть более длинного числа)
    re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)"),
)


def _valid(year: int, month: int, day: int,
           today: _dt.date | None = None) -> _dt.date | None:
    try:
        d = _dt.date(year, month, day)
    except ValueError:
        return None
    if d.year < _MIN_YEAR:
        return None
    today = today or _dt.date.today()
    if (d - today).days > _MAX_FUTURE_DAYS:
        return None
    return d


def parse_date_from_name(name: str, today: _dt.date | None = None) -> str | None:
    """Достать дату планёрки из имени файла. Возвращает 'YYYY-MM-DD' или None."""
    if not name:
        return None
    for i, pat in enumerate(_PATTERNS):
        for m in pat.finditer(name):
            if i == 1:  # DD.MM.YYYY — год в третьей группе
                day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            else:
                year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            d = _valid(year, month, day, today)
            if d:
                return d.isoformat()
    return None


def parse_explicit_date(value: str | None) -> str | None:
    """Проверить дату из формы/env. Принимает только ISO 'YYYY-MM-DD'."""
    if not value:
        return None
    value = value.strip()
    m = re.fullmatch(r"(20\d{2})-(\d{2})-(\d{2})", value)
    if not m:
        return None
    d = _valid(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return d.isoformat() if d else None


def resolve_meeting_date(explicit: str | None = None,
                         filename: str | None = None,
                         mtime: float | None = None) -> tuple[str | None, str]:
    """Определить дату встречи. Возвращает (date | None, source).

    source ∈ {"explicit", "filename", "mtime", "none"} — пишется в meta,
    чтобы в приёмке было видно, насколько дате можно верить.
    """
    d = parse_explicit_date(explicit)
    if d:
        return d, "explicit"
    d = parse_date_from_name(filename or "")
    if d:
        return d, "filename"
    if mtime:
        try:
            return _dt.date.fromtimestamp(mtime).isoformat(), "mtime"
        except (OSError, OverflowError, ValueError):
            pass
    return None, "none"
