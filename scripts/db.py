#!/usr/bin/env python3
"""
db.py — SQLite-слой состояния приложения (см. docs/adr/002).

Хранит ровно то, что НЕ является задачей (задачи живут в Trello, ADR-001):
  meeting          — метаданные встреч с настоящей датой
  proposal         — предложения ИИ и их судьба в приёмке
  dispatch_journal — журнал отправки в Trello; due_v1 неизменяем (триггер)
  speaker_map      — SPEAKER_XX → человек, на одну встречу

Файл БД — в named-томе Docker (/data), не в bind-mount: SQLite поверх
виндового bind-mount ловит проблемы блокировок. Бэкап — VACUUM INTO
в /output/backups (виден на хосте).

Потокобезопасность: соединение на операцию (get_db) + WAL. Нагрузка —
единицы запросов в минуту, этого достаточно.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

_DEFAULT_DB = "/data/transcriber.db"
_FALLBACK_DB = "/output/_state.db"     # если /data не смонтирован (CLI, dev)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meeting (
    id           INTEGER PRIMARY KEY,
    stem         TEXT NOT NULL,
    dir          TEXT NOT NULL,                 -- 'Категория/Подкатегория'
    content_hash TEXT,
    title        TEXT DEFAULT '',
    date         TEXT DEFAULT '',
    date_source  TEXT DEFAULT '',
    category     TEXT DEFAULT '',
    subcategory  TEXT DEFAULT '',
    audio        TEXT DEFAULT '',
    n_actions    INTEGER NOT NULL DEFAULT 0,
    n_decisions  INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT,
    UNIQUE (dir, stem)
);
CREATE INDEX IF NOT EXISTS idx_meeting_hash ON meeting(content_hash);

CREATE TABLE IF NOT EXISTS proposal (
    id              INTEGER PRIMARY KEY,
    meeting_id      INTEGER NOT NULL REFERENCES meeting(id) ON DELETE CASCADE,
    src_key         TEXT NOT NULL,              -- хэш исходного текста экстрактора:
                                                -- идентичность пункта между прогонами
    idx             INTEGER NOT NULL,           -- позиция в *_actions.json (порядок)
    kind            TEXT NOT NULL DEFAULT 'task',      -- task | minor
    text            TEXT NOT NULL,
    owner           TEXT,
    owner_matched   INTEGER NOT NULL DEFAULT 0,
    label           TEXT NOT NULL DEFAULT '',
    criterion       TEXT NOT NULL DEFAULT '',
    due             TEXT,                        -- YYYY-MM-DD, ставит человек
    status          TEXT NOT NULL DEFAULT 'proposed',
                        -- proposed | accepted | rejected | sent
    confidence      TEXT DEFAULT '',
    quote           TEXT DEFAULT '',
    quote_verified  INTEGER NOT NULL DEFAULT 0,
    t_sec           INTEGER,
    context         TEXT DEFAULT '',              -- контекст из извлечения
    section         TEXT DEFAULT '',              -- раздел/тема из извлечения
    source_deadline TEXT DEFAULT '',             -- срок словами из речи (подсветка)
    extractor       TEXT DEFAULT '',             -- какой моделью извлечено
    updated_at      TEXT,
    UNIQUE (meeting_id, src_key)
);

CREATE TABLE IF NOT EXISTS dispatch_journal (
    id             INTEGER PRIMARY KEY,
    sent_at        TEXT NOT NULL DEFAULT (datetime('now')),
    proposal_id    INTEGER REFERENCES proposal(id) ON DELETE SET NULL,
    meeting_id     INTEGER REFERENCES meeting(id) ON DELETE SET NULL,
    text           TEXT NOT NULL,
    owner          TEXT DEFAULT '',
    due_v1         TEXT,                         -- ПЕРВЫЙ срок, не меняется
    trello_card_id TEXT DEFAULT '',
    trello_url     TEXT DEFAULT '',
    t_sec          INTEGER,
    mirrored       INTEGER NOT NULL DEFAULT 0    -- зеркало в Telegram ушло
);

-- Вся ценность журнала — в неизменяемости первого срока.
CREATE TRIGGER IF NOT EXISTS trg_due_v1_immutable
BEFORE UPDATE ON dispatch_journal
FOR EACH ROW WHEN NEW.due_v1 IS NOT OLD.due_v1
BEGIN
    SELECT RAISE(ABORT, 'due_v1 is immutable');
END;

CREATE TABLE IF NOT EXISTS speaker_map (
    meeting_id    INTEGER NOT NULL REFERENCES meeting(id) ON DELETE CASCADE,
    speaker_label TEXT NOT NULL,
    person        TEXT NOT NULL,
    PRIMARY KEY (meeting_id, speaker_label)
);
"""


def db_path() -> Path:
    p = os.environ.get("DB_PATH")
    if p:
        return Path(p)
    if Path("/data").is_dir():
        return Path(_DEFAULT_DB)
    if Path("/output").is_dir():
        return Path(_FALLBACK_DB)
    return Path(__file__).resolve().parent.parent / "output" / "_state.db"


@contextmanager
def get_db(path: Path | None = None):
    p = path or db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db(path: Path | None = None) -> None:
    p = path or db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p, timeout=30)
    try:
        con.execute("PRAGMA journal_mode = WAL")
        con.execute("PRAGMA synchronous = NORMAL")
        con.executescript(_SCHEMA)
        con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        con.commit()
    finally:
        con.close()


# ─── meeting ─────────────────────────────────────────────────────────────────

def _sha256_file(path: Path, limit_mb: int = 64) -> str | None:
    """Хэш транскрипта — идентичность встречи при переобработке/переименовании."""
    try:
        h = hashlib.sha256()
        read = 0
        with open(path, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
                read += len(chunk)
                if read > limit_mb << 20:
                    break
        return h.hexdigest()
    except OSError:
        return None


def upsert_meeting(con: sqlite3.Connection, dir_rel: str, stem: str,
                   meta: dict, content_hash: str | None = None) -> int:
    """Идемпотентный upsert по (dir, stem). Возвращает meeting.id.

    Дата не затирается пустой: однажды определённая дата живёт, пока её
    явно не поменяли (meta с новой датой её обновит — это осознанная правка).
    """
    row = con.execute(
        "SELECT id FROM meeting WHERE dir = ? AND stem = ?", (dir_rel, stem)
    ).fetchone()
    fields = {
        "title":       meta.get("title") or stem,
        "category":    meta.get("category") or "",
        "subcategory": meta.get("subcategory") or "",
        "audio":       meta.get("audio") or "",
        "updated_at":  time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if meta.get("date"):
        fields["date"] = meta["date"]
        fields["date_source"] = meta.get("date_source") or ""
    if content_hash:
        fields["content_hash"] = content_hash
    if row:
        sets = ", ".join(f"{k} = ?" for k in fields)
        con.execute(f"UPDATE meeting SET {sets} WHERE id = ?",
                    (*fields.values(), row["id"]))
        return int(row["id"])
    cols = ["dir", "stem", *fields.keys()]
    q = ", ".join("?" for _ in cols)
    cur = con.execute(
        f"INSERT INTO meeting ({', '.join(cols)}) VALUES ({q})",
        (dir_rel, stem, *fields.values()),
    )
    return int(cur.lastrowid)


def sync_meetings_from_output(output_dir: Path) -> dict:
    """Импорт/обновление встреч из файлов output/ (истина — на диске).

    Безопасно гонять при каждом старте: upsert, ничего не удаляет.
    """
    stats = {"seen": 0, "imported": 0}
    if not output_dir.is_dir():
        return stats
    with get_db() as con:
        for meta_path in output_dir.rglob("*_meta.json"):
            rel_parts = meta_path.relative_to(output_dir).parts
            if rel_parts and rel_parts[0] in ("_obsidian", "backups"):
                continue
            if "intermediate" in rel_parts:
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if not isinstance(meta, dict):
                    continue
            except Exception:
                continue
            stem = meta.get("stem") or meta_path.name[: -len("_meta.json")]
            dir_rel = meta_path.parent.relative_to(output_dir).as_posix()
            # Категория/подкатегория — из пути (истина): старые меты писали
            # в category имя последнего сегмента («Без подкатегории»).
            parts = [p for p in dir_rel.split("/") if p]
            if parts:
                meta = {**meta,
                        "category": parts[0],
                        "subcategory": parts[1] if len(parts) > 1 else ""}
            speakers = meta_path.parent / f"{stem}_speakers.txt"
            chash = _sha256_file(speakers) if speakers.exists() else None
            stats["seen"] += 1
            before = con.total_changes
            mid = upsert_meeting(con, dir_rel, stem, meta, chash)
            if con.total_changes > before:
                stats["imported"] += 1
            actions_path = meta_path.parent / f"{stem}_actions.json"
            n_actions = 0
            if actions_path.exists():
                try:
                    actions = json.loads(actions_path.read_text(encoding="utf-8"))
                    if isinstance(actions, list):
                        n_actions = len(actions)
                        sync_proposals(con, mid, actions)
                except Exception:
                    pass
            n_decisions = 0
            dec_path = meta_path.parent / f"{stem}_decisions.json"
            if dec_path.exists():
                try:
                    dec = json.loads(dec_path.read_text(encoding="utf-8"))
                    if isinstance(dec, list):
                        n_decisions = len(dec)
                except Exception:
                    pass
            con.execute("UPDATE meeting SET n_actions = ?, n_decisions = ? WHERE id = ?",
                        (n_actions, n_decisions, mid))
    return stats


# ─── proposals (приёмка) ─────────────────────────────────────────────────────

_PROPOSAL_STATUSES = ("proposed", "accepted", "rejected", "sent")

# Поля, которыми владеет экстрактор. Обновляются, только пока строка
# в статусе 'proposed' — правки человека переобработка не затирает.
_EXTRACTOR_FIELDS = ("kind", "text", "owner", "owner_matched", "confidence",
                     "quote", "quote_verified", "t_sec", "context",
                     "source_deadline", "extractor")


def _action_to_fields(a: dict) -> dict:
    owner = a.get("owner")
    # kind нет в старых actions.json → считаем по правилу ТЗ:
    # исполнитель есть И уверенность high → задача, иначе мелочь.
    fallback_kind = ("task" if owner and str(a.get("confidence", "")).lower() == "high"
                     else "minor")
    return {
        "kind":            a.get("kind") or fallback_kind,
        "text":            (a.get("task") or a.get("text") or "").strip(),
        "owner":           owner if owner else None,
        "owner_matched":   int(bool(a.get("owner_matched"))),
        "confidence":      (a.get("confidence") or "").lower(),
        "quote":           a.get("evidence") or a.get("quote") or "",
        "quote_verified":  int(bool(a.get("quote_verified"))),
        "t_sec":           a.get("t_sec"),
        "context":         a.get("context") or "",
        "section":         a.get("section") or "",
        "source_deadline": a.get("deadline") or "",
        "extractor":       a.get("extractor") or "",
    }


def _src_key(a: dict) -> str:
    """Идентичность пункта между прогонами — нормализованный текст экстрактора.

    Позиционный idx для этого не годится: человек уже отправил пункт №0,
    переобработка сдвинула нумерацию — и новый пункт на месте отправленного
    молча терялся бы.
    """
    base = (a.get("task") or a.get("text") or "")
    norm = re.sub(r"\s+", " ", str(base)).strip().lower().replace("ё", "е")
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def sync_proposals(con: sqlite3.Connection, meeting_id: int,
                   actions: list[dict]) -> None:
    """Синхронизировать предложения из *_actions.json.

    Идемпотентно по (meeting_id, src_key). Правки человека святы: строки со
    статусом ≠ 'proposed' не трогаем. Proposed-строки, которых больше нет
    в выгрузке экстрактора, удаляем; переформулированный моделью пункт
    приходит как новый (человек его примет или отклонит).
    """
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    seen: list[str] = []
    for idx, a in enumerate(actions):
        if not isinstance(a, dict):
            continue
        fields = _action_to_fields(a)
        if not fields["text"]:
            continue
        key = _src_key(a)
        if key in seen:            # дубль внутри одной выгрузки
            continue
        seen.append(key)
        row = con.execute(
            "SELECT id, status FROM proposal WHERE meeting_id = ? AND src_key = ?",
            (meeting_id, key),
        ).fetchone()
        if row is None:
            cols = ["meeting_id", "src_key", "idx", "status", "updated_at",
                    *fields.keys()]
            q = ", ".join("?" for _ in cols)
            con.execute(
                f"INSERT INTO proposal ({', '.join(cols)}) VALUES ({q})",
                (meeting_id, key, idx, "proposed", now, *fields.values()),
            )
        elif row["status"] == "proposed":
            sets = ", ".join(f"{k} = ?" for k in fields)
            con.execute(
                f"UPDATE proposal SET {sets}, idx = ?, updated_at = ? WHERE id = ?",
                (*fields.values(), idx, now, row["id"]),
            )
    if seen:
        marks = ", ".join("?" for _ in seen)
        con.execute(
            f"DELETE FROM proposal WHERE meeting_id = ? AND status = 'proposed' "
            f"AND src_key NOT IN ({marks})",
            (meeting_id, *seen),
        )
    else:
        con.execute(
            "DELETE FROM proposal WHERE meeting_id = ? AND status = 'proposed'",
            (meeting_id,),
        )


def update_proposal(con: sqlite3.Connection, proposal_id: int,
                    patch: dict) -> dict | None:
    """Правка полей приёмки человеком. Возвращает обновлённую строку."""
    allowed = {"text", "owner", "label", "criterion", "due", "status", "kind"}
    fields = {}
    for k, v in patch.items():
        if k not in allowed:
            continue
        if k == "status" and v not in _PROPOSAL_STATUSES:
            continue
        fields[k] = v
    if fields:
        sets = ", ".join(f"{k} = ?" for k in fields)
        con.execute(
            f"UPDATE proposal SET {sets}, updated_at = ? WHERE id = ?",
            (*fields.values(), time.strftime("%Y-%m-%d %H:%M:%S"), proposal_id),
        )
    row = con.execute("SELECT * FROM proposal WHERE id = ?",
                      (proposal_id,)).fetchone()
    return dict(row) if row else None


def proposals_for_meeting(con: sqlite3.Connection, meeting_id: int) -> list[dict]:
    rows = con.execute(
        "SELECT * FROM proposal WHERE meeting_id = ? ORDER BY idx", (meeting_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def inbox_meetings(con: sqlite3.Connection, limit: int = 30) -> list[dict]:
    """Встречи для приёмки: есть предложения, новее — выше."""
    rows = con.execute(
        """
        SELECT m.*, COUNT(p.id) AS n_total,
               SUM(CASE WHEN p.status = 'proposed' AND p.kind = 'task'
                        THEN 1 ELSE 0 END) AS n_pending
        FROM meeting m JOIN proposal p ON p.meeting_id = m.id
        GROUP BY m.id
        ORDER BY m.date DESC, m.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def set_speaker(con: sqlite3.Connection, meeting_id: int,
                label: str, person: str) -> None:
    con.execute(
        "INSERT INTO speaker_map (meeting_id, speaker_label, person) VALUES (?, ?, ?) "
        "ON CONFLICT(meeting_id, speaker_label) DO UPDATE SET person = excluded.person",
        (meeting_id, label, person),
    )
    # Владельцы-метки в ещё не отправленных предложениях → человек
    con.execute(
        "UPDATE proposal SET owner = ?, owner_matched = 1 "
        "WHERE meeting_id = ? AND owner = ? AND status IN ('proposed', 'accepted')",
        (person, meeting_id, label),
    )


# ─── journal ─────────────────────────────────────────────────────────────────

def journal_append(con: sqlite3.Connection, *, text: str, owner: str,
                   due_v1: str | None, meeting_id: int | None,
                   proposal_id: int | None, t_sec: int | None,
                   trello_card_id: str = "", trello_url: str = "") -> int:
    cur = con.execute(
        "INSERT INTO dispatch_journal "
        "(text, owner, due_v1, meeting_id, proposal_id, t_sec, trello_card_id, trello_url) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (text, owner or "", due_v1, meeting_id, proposal_id, t_sec,
         trello_card_id, trello_url),
    )
    return int(cur.lastrowid)


# ─── аналитика: Брифинг / Расписание / Пульс ─────────────────────────────────

_LABEL_RE = re.compile(r"^(SPEAKER|PERSON|UNKNOWN)[_ ]?", re.IGNORECASE)


def meetings_all(con: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in con.execute(
        "SELECT * FROM meeting WHERE date != '' ORDER BY date DESC, id DESC")]


def series_stats(con: sqlite3.Connection) -> list[dict]:
    """Серии = категории; ритм посчитан по реальным записям, не по расписанию."""
    import datetime as _dt
    groups: dict[str, list[dict]] = {}
    for m in meetings_all(con):
        groups.setdefault(m["category"] or "Общее", []).append(m)
    today = _dt.date.today()
    res = []
    for s, ms in groups.items():
        dates = sorted(m["date"] for m in ms)
        try:
            span = max(1, (_dt.date.fromisoformat(dates[-1])
                           - _dt.date.fromisoformat(dates[0])).days)
            days_since = (today - _dt.date.fromisoformat(dates[-1])).days
        except ValueError:
            continue
        per_week = round(len(ms) / (span / 7), 1) if len(ms) > 1 else 1.0 * len(ms)
        res.append({
            "series": s, "n": len(ms),
            "last_date": dates[-1], "days_since": days_since,
            "per_week": per_week,
            "n_actions": sum(m["n_actions"] or 0 for m in ms),
            "n_decisions": sum(m["n_decisions"] or 0 for m in ms),
        })
    res.sort(key=lambda x: x["last_date"], reverse=True)
    return res


def open_tasks_for_series(con: sqlite3.Connection, category: str,
                          limit: int = 12) -> list[dict]:
    """Задачи серии, ещё не ушедшие в Trello (kind=task, не отправлены)."""
    rows = con.execute(
        """
        SELECT p.*, m.date AS m_date, m.title AS m_title
        FROM proposal p JOIN meeting m ON m.id = p.meeting_id
        WHERE m.category = ? AND p.kind = 'task'
          AND p.status IN ('proposed', 'accepted')
        ORDER BY m.date DESC, p.idx
        LIMIT ?
        """,
        (category, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def pulse_summary(con: sqlite3.Connection, days: int = 90) -> dict:
    """Данные Пульса из SQLite; Trello не нужен (сверка — отдельной кнопкой)."""
    import datetime as _dt
    from collections import Counter

    today = _dt.date.today()
    since = (today - _dt.timedelta(days=days)).isoformat()
    ms = [m for m in meetings_all(con) if m["date"] >= since]

    mids = [m["id"] for m in ms]
    props: list[dict] = []
    if mids:
        marks = ", ".join("?" for _ in mids)
        props = [dict(r) for r in con.execute(
            f"SELECT * FROM proposal WHERE meeting_id IN ({marks})", mids)]
    tasks = [p for p in props if p["kind"] == "task"]

    def _is_label(o: str | None) -> bool:
        return bool(o) and bool(_LABEL_RE.match(o))

    no_owner = sum(1 for p in tasks if not p["owner"] or _is_label(p["owner"]))
    by_owner = Counter(p["owner"] for p in tasks
                       if p["owner"] and not _is_label(p["owner"]))
    by_section = Counter((p["section"] or "").strip() for p in props
                         if (p["section"] or "").strip())

    weeks = [0] * 12
    for m in ms:
        try:
            wk = (today - _dt.date.fromisoformat(m["date"])).days // 7
        except ValueError:
            continue
        if 0 <= wk < 12:
            weeks[11 - wk] += 1

    journal = [dict(r) for r in con.execute(
        "SELECT * FROM dispatch_journal ORDER BY id DESC LIMIT 10")]

    return {
        "days": days,
        "n_meetings": len(ms),
        "n_actions": sum(m["n_actions"] or 0 for m in ms),
        "n_decisions": sum(m["n_decisions"] or 0 for m in ms),
        "n_tasks": len(tasks),
        "no_owner": no_owner,
        "by_owner": by_owner.most_common(8),
        "by_section": by_section.most_common(8),
        "weeks": weeks,
        "silent": [s for s in series_stats(con) if s["days_since"] > 14],
        "journal": journal,
    }


def journal_with_cards(con: sqlite3.Connection, limit: int = 50) -> list[dict]:
    rows = con.execute(
        "SELECT * FROM dispatch_journal WHERE trello_card_id != '' "
        "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


# ─── backup ──────────────────────────────────────────────────────────────────

def backup(backups_dir: Path, keep: int = 14) -> Path | None:
    """VACUUM INTO копию с меткой времени; держим последние `keep` штук."""
    src = db_path()
    if not src.exists():
        return None
    backups_dir.mkdir(parents=True, exist_ok=True)
    dest = backups_dir / f"transcriber-{time.strftime('%Y%m%d-%H%M%S')}.db"
    con = sqlite3.connect(src, timeout=30)
    try:
        con.execute("VACUUM INTO ?", (str(dest),))
    finally:
        con.close()
    old = sorted(backups_dir.glob("transcriber-*.db"))
    for f in old[:-keep]:
        try:
            f.unlink()
        except OSError:
            pass
    return dest
