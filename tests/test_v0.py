#!/usr/bin/env python3
"""
Тесты v0 («задачи со сроками», docs/tz-v0.md). Без внешних зависимостей:
  python -m unittest discover tests -v

Покрывают приёмочные критерии шагов:
  1 — дата встречи из имени файла;
  2 — нормализация владельцев из падежей;
  2б — якорный поиск цитат + таймкоды кодом + токены вне карты;
  3/4 — идемпотентность предложений, неизменяемость due_v1, правки человека
        переживают переобработку.
"""

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from meeting_date import parse_date_from_name, parse_explicit_date  # noqa: E402
from people import normalize_owner  # noqa: E402
from evidence import verify_item, unresolved_tokens  # noqa: E402


class TestMeetingDate(unittest.TestCase):
    def test_from_filename(self):
        cases = {
            "bandicam 2026-06-22 11-52-27-648": "2026-06-22",
            "планёрка 22.06.2026 вечер":        "2026-06-22",
            "20260622_planerka":                "2026-06-22",
            "rec_2026_08_03":                   "2026-08-03",
        }
        for name, want in cases.items():
            self.assertEqual(parse_date_from_name(name), want, name)

    def test_garbage_rejected(self):
        for name in ("Crm system audi", "meeting 2026-13-45", "call 99.99.2026",
                     "backup 2019-05-01", "план 2030-01-01"):
            self.assertIsNone(parse_date_from_name(name), name)

    def test_explicit(self):
        self.assertEqual(parse_explicit_date("2026-08-06"), "2026-08-06")
        self.assertIsNone(parse_explicit_date("06.08.2026"))
        self.assertIsNone(parse_explicit_date("2031-01-01"))


class TestOwners(unittest.TestCase):
    PEOPLE = ["Никита", "Денис", "Влад", "Мария", "Даша", "Ален", "Дизайнер"]

    def test_cases_normalize(self):
        for raw, want in [("Владу", "Влад"), ("Марии", "Мария"),
                          ("Денису", "Денис"), ("Дизайнеров", "Дизайнер"),
                          ("Алён", "Ален")]:
            got, matched = normalize_owner(raw, self.PEOPLE)
            self.assertTrue(matched, raw)
            self.assertEqual(got, want, raw)

    def test_labels_and_garbage(self):
        self.assertEqual(normalize_owner("SPEAKER_00", self.PEOPLE),
                         ("SPEAKER_00", False))
        self.assertEqual(normalize_owner("не назначен", self.PEOPLE),
                         (None, False))
        got, matched = normalize_owner("Ладно", self.PEOPLE)
        self.assertFalse(matched)


class TestEvidence(unittest.TestCase):
    TRANSCRIPT = (
        "[00:00:01 - 00:00:08] SPEAKER_00:\n"
        "Всем привет, давайте начнём. Сегодня про инбокс.\n\n"
        "[00:01:30 - 00:03:00] SPEAKER_01:\n"
        "Мне зафиксировано, я же сказал, что мультиаккаунт мы делаем "
        "после доступов.\n"
    )

    def test_model_fixed_asr_still_found(self):
        ok, t = verify_item(
            "Я зафиксировал, я же сказал, что мультиаккаунт мы делаем "
            "после доступов", self.TRANSCRIPT)
        self.assertTrue(ok)
        self.assertEqual(t, 90)

    def test_fabricated_rejected(self):
        ok, t = verify_item("Мы решили купить трактор", self.TRANSCRIPT)
        self.assertFalse(ok)
        self.assertIsNone(t)

    def test_unresolved_tokens(self):
        got = unresolved_tokens("Ответственный PERSON_014, компания COMPANY_002",
                                {"COMPANY_002"})
        self.assertEqual(got, {"PERSON_014"})


class TestStateFlow(unittest.TestCase):
    """Полный цикл: встреча → предложения → правки человека → отправка."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["DB_PATH"] = str(Path(self.tmp) / "state.db")
        global db
        import db  # noqa: F401  (после установки DB_PATH)
        import importlib
        importlib.reload(db)
        db.init_db()
        self.db = db

    def _actions(self):
        return [
            {"task": "Сделать мультиаккаунт", "owner": "Денис",
             "confidence": "high", "evidence": "как получу доступы — два дня",
             "kind": "task", "quote_verified": True, "t_sec": 130,
             "deadline": "два дня", "extractor": "test-model"},
            {"task": "Надо бы посмотреть баннеры", "owner": None,
             "confidence": "medium", "kind": "minor"},
        ]

    def test_full_flow(self):
        d = self.db
        with d.get_db() as con:
            mid = d.upsert_meeting(con, "Ежедневки/Тест", "rec 2026-08-05",
                                   {"title": "Планёрка", "date": "2026-08-05"})
            d.sync_proposals(con, mid, self._actions())
            props = d.proposals_for_meeting(con, mid)
            self.assertEqual(len(props), 2)
            task = next(p for p in props if p["kind"] == "task")

            # человек правит: критерий + срок + принял
            d.update_proposal(con, task["id"], {
                "criterion": "два аккаунта, переписка не путается",
                "due": "2026-08-08", "status": "accepted"})

            # переобработка: extractor поменял текст — правки человека святы
            changed = self._actions()
            changed[0]["task"] = "ПЕРЕПИСАННЫЙ ТЕКСТ ОТ МОДЕЛИ"
            d.sync_proposals(con, mid, changed)
            row = dict(con.execute("SELECT * FROM proposal WHERE id = ?",
                                   (task["id"],)).fetchone())
            self.assertEqual(row["text"], "Сделать мультиаккаунт")
            self.assertEqual(row["due"], "2026-08-08")
            self.assertEqual(row["status"], "accepted")

            # отправка: журнал + статус sent
            jid = d.journal_append(con, text=row["text"], owner=row["owner"],
                                   due_v1=row["due"], meeting_id=mid,
                                   proposal_id=row["id"], t_sec=row["t_sec"],
                                   trello_card_id="tc_1", trello_url="https://trello/x")
            con.execute("UPDATE proposal SET status = 'sent' WHERE id = ?",
                        (row["id"],))

            # due_v1 неизменяем
            with self.assertRaises(sqlite3.IntegrityError):
                con.execute("UPDATE dispatch_journal SET due_v1 = '2026-09-01' "
                            "WHERE id = ?", (jid,))

        # имена спикеров: владельцы-метки в неотправленных обновляются
        with d.get_db() as con:
            d.sync_proposals(con, mid, [
                {"task": "Проверить провайдера", "owner": "SPEAKER_00",
                 "confidence": "high", "kind": "task"}])
            d.set_speaker(con, mid, "SPEAKER_00", "Никита")
            owners = [p["owner"] for p in d.proposals_for_meeting(con, mid)
                      if p["text"] == "Проверить провайдера"]
            self.assertEqual(owners, ["Никита"])

    def test_pulse_and_series(self):
        import datetime
        d = self.db
        today = datetime.date.today()
        d1 = (today - datetime.timedelta(days=2)).isoformat()
        d2 = (today - datetime.timedelta(days=9)).isoformat()
        d3 = (today - datetime.timedelta(days=30)).isoformat()
        with d.get_db() as con:
            for i, (dt_, cat) in enumerate([(d1, "Ежедневки"), (d2, "Ежедневки"),
                                            (d3, "Контент")]):
                mid = d.upsert_meeting(con, f"{cat}/Общее", f"m{i}",
                                       {"title": f"Встреча {i}", "date": dt_,
                                        "category": cat})
                con.execute("UPDATE meeting SET n_actions=5, n_decisions=2 "
                            "WHERE id=?", (mid,))
                d.sync_proposals(con, mid, [
                    {"task": f"Задача {i}", "owner": "Денис",
                     "confidence": "high", "kind": "task", "section": "CRM"},
                    {"task": f"Мелочь {i}", "owner": None, "kind": "minor"}])
            series = d.series_stats(con)
            self.assertEqual({s["series"] for s in series},
                             {"Ежедневки", "Контент"})
            silent = [s for s in series if s["days_since"] > 14]
            self.assertEqual([s["series"] for s in silent], ["Контент"])

            pulse = d.pulse_summary(con, days=90)
            self.assertEqual(pulse["n_meetings"], 3)
            self.assertEqual(pulse["n_actions"], 15)
            self.assertEqual(pulse["by_owner"][0][0], "Денис")
            self.assertEqual(pulse["by_section"][0], ("CRM", 3))
            self.assertEqual(sum(pulse["weeks"]), 3)

            open_tasks = d.open_tasks_for_series(con, "Ежедневки")
            self.assertEqual(len(open_tasks), 2)
            self.assertTrue(all(t["kind"] == "task" for t in open_tasks))


if __name__ == "__main__":
    unittest.main()
