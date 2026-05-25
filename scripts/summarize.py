#!/usr/bin/env python3
"""
summarize.py — Meeting Intelligence Pipeline (evidence-first)

Flow:
    *_speakers.txt
    → apply speaker_map.json
    → parse speaker blocks
    → build chunks with overlap (по границам speaker-блоков)
    → MAP: structured JSON extraction per chunk  → save intermediate/
    → MERGE all chunk JSONs
    → DEDUPLICATE (по Jaccard similarity)
    → EVIDENCE CHECK (downgrade confidence if missing)
    → save *_analysis.json
    → REDUCE: generate final summary from structured data
    → save *_summary.md, *_actions.json, *_decisions.json

Env:
    LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
    CHUNK_SIZE_CHARS    (default 12000)
    CHUNK_OVERLAP_CHARS (default 2000)
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] pip install openai", file=sys.stderr)
    sys.exit(1)

# Optional local anonymization (anonymize.py in same directory)
sys.path.insert(0, str(Path(__file__).parent))
try:
    from anonymize import Anonymizer as _Anonymizer
    _ANON_AVAILABLE = True
except ImportError:
    _ANON_AVAILABLE = False

# ─── Config ───────────────────────────────────────────────────────────────────

CHUNK_SIZE_CHARS    = int(os.environ.get("CHUNK_SIZE_CHARS",    "12000"))
CHUNK_OVERLAP_CHARS = int(os.environ.get("CHUNK_OVERLAP_CHARS", "2000"))

# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class SpeakerBlock:
    speaker:    str
    start_time: str   # "HH:MM:SS"
    end_time:   str   # "HH:MM:SS"
    text:       str

    def to_text(self) -> str:
        return f"[{self.start_time} - {self.end_time}] {self.speaker}:\n{self.text}"

    def char_len(self) -> int:
        return len(self.to_text())


# ─── LLM ──────────────────────────────────────────────────────────────────────

def _get_llm_client() -> tuple["OpenAI", str]:
    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
    api_key  = os.environ.get("LLM_API_KEY",  "not-required")
    model    = os.environ.get("LLM_MODEL",     "qwen3-32b")

    # In Docker: localhost → host.docker.internal
    if os.path.exists("/.dockerenv") and re.search(r"localhost|127\.0\.0\.1", base_url):
        base_url = re.sub(r"localhost|127\.0\.0\.1", "host.docker.internal", base_url)
        print(f"[INFO] Docker detected: LLM_BASE_URL → {base_url}")

    return OpenAI(base_url=base_url, api_key=api_key), model


def _chat(client: "OpenAI", model: str, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def _clean_llm_response(raw: str) -> str:
    """Strip <think>…</think> blocks (Qwen3 thinking mode) and markdown fences."""
    raw = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
        if m:
            return m.group(1).strip()
    return raw


# ─── Parsing ──────────────────────────────────────────────────────────────────

_BLOCK_RE = re.compile(
    r"\[(\d{2}:\d{2}:\d{2}) - (\d{2}:\d{2}:\d{2})\] ([^\n:]+):\n(.*?)(?=\n\[\d{2}:\d{2}|\Z)",
    re.DOTALL,
)


def parse_speaker_blocks(transcript: str) -> list[SpeakerBlock]:
    blocks = []
    for m in _BLOCK_RE.finditer(transcript):
        blocks.append(SpeakerBlock(
            start_time = m.group(1),
            end_time   = m.group(2),
            speaker    = m.group(3).strip(),
            text       = m.group(4).strip(),
        ))
    return blocks


def _apply_speaker_map(text: str, speaker_map: dict) -> str:
    for code, name in speaker_map.items():
        text = text.replace(code, name)
    return text


# ─── Chunking with overlap ────────────────────────────────────────────────────

def build_chunks_with_overlap(
    blocks: list[SpeakerBlock],
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap_size: int = CHUNK_OVERLAP_CHARS,
) -> list[list[SpeakerBlock]]:
    """
    Split speaker blocks into chunks of ~chunk_size chars with overlap.
    Never cuts mid-block. Overlap is measured in chars, aligned to block boundaries.
    """
    if not blocks:
        return []

    chunks: list[list[SpeakerBlock]] = []
    start_idx = 0

    while start_idx < len(blocks):
        current: list[SpeakerBlock] = []
        current_len = 0
        idx = start_idx

        while idx < len(blocks):
            block = blocks[idx]
            bl = block.char_len()
            if current and current_len + bl > chunk_size:
                break
            current.append(block)
            current_len += bl
            idx += 1

        # Single block larger than chunk_size — include it anyway to avoid infinite loop
        if not current:
            current = [blocks[start_idx]]
            idx = start_idx + 1

        chunks.append(current)

        if idx >= len(blocks):
            break

        # Find next chunk start: walk backward from idx until we've covered overlap_size chars
        overlap_chars = 0
        next_start = idx
        while next_start > start_idx + 1:
            overlap_chars += blocks[next_start - 1].char_len()
            next_start -= 1
            if overlap_chars >= overlap_size:
                break

        # Safety: always advance by at least 1 block
        if next_start <= start_idx:
            next_start = start_idx + max(1, len(current) // 2)

        start_idx = next_start

    return chunks


def chunk_to_text(chunk: list[SpeakerBlock]) -> str:
    return "\n\n".join(b.to_text() for b in chunk)


# ─── MAP: structured extraction ───────────────────────────────────────────────

_MAP_SYSTEM = """\
Ты анализируешь фрагмент транскрипта рабочей планёрки.
Твоя задача — извлечь МАКСИМУМ структурированных фактов. Это НЕ краткое резюме.

КРИТИЧЕСКИ ВАЖНО: на продуктовых планёрках люди часто идут по списку разделов/модулей системы и для каждого фиксируют статус, приоритет, проблему. Эти статусы — главное содержание встречи. Извлекай их в section_statuses, даже если люди не говорят формальное "статус такой-то".

Также фиксируй НЕЯВНЫЕ блокеры, риски и открытые вопросы — если человек жалуется на проблему или сомневается, это блокер/риск/вопрос, даже без слова "блокер".

Верни ТОЛЬКО строго валидный JSON без Markdown и без пояснений. Схема:

{
  "chunk_id": <число>,
  "time_range": {"start": "HH:MM:SS", "end": "HH:MM:SS"},
  "summary": "5-10 предложений: какие разделы/темы обсуждались по порядку, какие статусы и приоритеты зафиксированы, какая бизнес-логика проявилась. Не пиши абстрактно 'обсуждали дизайн' — пиши конкретно 'обсудили раздел X, статус Y, потому что Z'.",

  "section_statuses": [
    {
      "section": "название раздела/модуля/функции (напр. 'Ценообразование', 'Панель сборщика', 'Клиентская часть')",
      "status": "missing|exists|in_progress|needs_rework|postponed|removed|exists_partial",
      "priority": "high|medium|low|none",
      "reasoning": "почему такой статус/приоритет — деталями из обсуждения",
      "notes": "что конкретно нужно сделать или что обсуждалось",
      "source_time": "HH:MM:SS или null",
      "evidence": "короткая цитата"
    }
  ],

  "sequence": [
    {
      "order": <число, порядок>,
      "what": "что делается/будет делаться",
      "depends_on": "от чего зависит или null",
      "timing": "когда (в разработке / следующее / после X / не приоритет / null)",
      "source_time": "HH:MM:SS или null"
    }
  ],

  "business_context": [
    {
      "topic": "тема (напр. 'Telegram-first', 'Доступные склады клиенту', 'Бесшовное переключение сборщик/курьер')",
      "why_it_matters": "зачем это нужно бизнесу/пользователю",
      "details": "конкретика, ограничения, варианты",
      "source_time": "HH:MM:SS или null",
      "evidence": "короткая цитата"
    }
  ],

  "decisions": [
    {
      "decision": "что решили",
      "context": "почему это решили",
      "speakers": ["имя"],
      "source_time": "HH:MM:SS или null",
      "evidence": "короткая цитата из transcript",
      "confidence": "high|medium|low"
    }
  ],
  "action_items": [
    {
      "task": "что нужно сделать — конкретно, не 'правки в дизайне', а 'добавить инвентаризацию в панель сборщика'",
      "owner": "имя или null",
      "deadline": "дата или null",
      "context": "контекст поручения — почему и в связи с чем",
      "section": "к какому разделу относится или null",
      "source_time": "HH:MM:SS или null",
      "evidence": "короткая цитата",
      "confidence": "high|medium|low"
    }
  ],
  "blockers": [
    {
      "blocker": "описание (включая НЕЯВНЫЕ: 'раздел отсутствует', 'дизайна не было', 'функционал сделан слишком поверхностно')",
      "impact": "влияние на работу",
      "owner": "имя или null",
      "inferred": false,
      "source_time": "HH:MM:SS или null",
      "evidence": "короткая цитата"
    }
  ],
  "risks": [
    {
      "risk": "описание (включая НЕЯВНЫЕ: 'риск переплатить за переделки', 'риск перегрузить мобильный экран')",
      "severity": "high|medium|low",
      "context": "контекст",
      "inferred": false,
      "source_time": "HH:MM:SS или null",
      "evidence": "короткая цитата"
    }
  ],
  "open_questions": [
    {
      "question": "вопрос (если по теме нет финального решения или решение отложено — это открытый вопрос)",
      "owner": "имя или null",
      "context": "контекст",
      "source_time": "HH:MM:SS или null"
    }
  ],
  "important_facts": ["факт"],
  "mentioned_topics": ["тема"]
}

Правила:
- Не выдумывай факты, которых нет в тексте.
- Если owner неизвестен — owner = null. Дедлайн неизвестен — deadline = null.
- Указывай source_time и evidence если возможно.
- decisions = только финальные решения. Обсуждение без решения → open_questions.
- НО: section_statuses, blockers, risks записывай ЩЕДРО — это главная ценность.
- Если раздел упомянут со статусом «есть/нет/в разработке/переделать/отложить/не приоритет» — обязательно в section_statuses.
- Если в обсуждении проскочила бизнес-логика (как должно работать, для кого, в каком контексте) — обязательно в business_context.
- В blockers/risks ставь `"inferred": true` если проблема явно не названа словом «блокер»/«риск», а выведена из контекста; иначе `false`.
- Сохраняй имена спикеров как они есть в transcript.
- В тексте могут встречаться токены вида PERSON_001, COMPANY_001, LOCATION_001, EMAIL_001, PHONE_001, DOMAIN_001, URL_001, TRANSACTION_001 — это анонимизированные реальные сущности. Обращайся с ними как с обычными именами собственными.\
"""


def extract_chunk(
    client: "OpenAI",
    model: str,
    chunk: list[SpeakerBlock],
    chunk_id: int,
    total_chunks: int,
    anonymizer=None,
) -> dict:
    time_start = chunk[0].start_time
    time_end   = chunk[-1].end_time
    text       = chunk_to_text(chunk)

    # ── LOCAL ANONYMIZATION before sending to online LLM ─────────────────────
    if anonymizer is not None:
        text = anonymizer.anonymize(text)

    raw = _chat(
        client, model, _MAP_SYSTEM,
        f"Фрагмент {chunk_id} из {total_chunks} [{time_start} — {time_end}]:\n\n{text}",
    )

    cleaned = _clean_llm_response(raw)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"[WARN] Chunk {chunk_id}: JSON parse failed, storing raw", file=sys.stderr)
        result = {
            "chunk_id": chunk_id,
            "time_range": {"start": time_start, "end": time_end},
            "summary": raw[:500],
            "decisions": [], "action_items": [], "blockers": [],
            "risks": [], "open_questions": [], "important_facts": [],
            "mentioned_topics": [], "_parse_error": True,
        }

    # ── LOCAL DE-ANONYMIZATION of LLM response ────────────────────────────────
    if anonymizer is not None:
        result = anonymizer.deanonymize_any(result)

    result["chunk_id"] = chunk_id
    result.setdefault("time_range", {"start": time_start, "end": time_end})
    return result


# ─── MERGE ────────────────────────────────────────────────────────────────────

def merge_chunks(chunk_results: list[dict]) -> dict:
    merged: dict = {
        "chunk_summaries":   [],
        "section_statuses":  [],
        "sequence":          [],
        "business_context":  [],
        "decisions":         [],
        "action_items":      [],
        "blockers":          [],
        "risks":              [],
        "open_questions":    [],
        "important_facts":   [],
        "mentioned_topics":  [],
    }
    for cr in chunk_results:
        if cr.get("summary"):
            merged["chunk_summaries"].append({
                "chunk_id":   cr.get("chunk_id"),
                "time_range": cr.get("time_range", {}),
                "summary":    cr["summary"],
            })
        for key in ("section_statuses", "sequence", "business_context",
                    "decisions", "action_items", "blockers", "risks",
                    "open_questions", "important_facts", "mentioned_topics"):
            items = cr.get(key, [])
            if isinstance(items, list):
                merged[key].extend(items)
    return merged


# ─── DEDUPLICATION ────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", str(text).lower()).strip()


def _jaccard(a: str, b: str) -> float:
    sa = set(_normalize(a).split())
    sb = set(_normalize(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _dedup_by_key(items: list[dict], key: str, threshold: float = 0.6) -> list[dict]:
    seen: list[str] = []
    result: list[dict] = []
    for item in items:
        text = item.get(key, "")
        if not any(_jaccard(text, s) >= threshold for s in seen):
            seen.append(text)
            result.append(item)
    return result


def _merge_section_statuses(items: list[dict]) -> list[dict]:
    """Combine multiple status entries for the same section (last non-empty wins, notes accumulate)."""
    bucket: dict[str, dict] = {}
    for it in items:
        key = _normalize(it.get("section", ""))
        if not key:
            continue
        prev = bucket.get(key)
        if prev is None:
            bucket[key] = dict(it)
            continue
        # Merge: prefer non-empty status/priority; concatenate notes/reasoning
        for fld in ("status", "priority", "source_time", "evidence"):
            if not prev.get(fld) and it.get(fld):
                prev[fld] = it[fld]
        for fld in ("reasoning", "notes"):
            old, new = prev.get(fld) or "", it.get(fld) or ""
            if new and new not in old:
                prev[fld] = (old + " | " + new).strip(" |") if old else new
    return list(bucket.values())


def deduplicate(merged: dict) -> dict:
    merged["section_statuses"] = _merge_section_statuses(merged.get("section_statuses", []))
    merged["business_context"] = _dedup_by_key(merged.get("business_context", []), "topic")
    merged["sequence"]         = _dedup_by_key(merged.get("sequence", []),         "what")
    merged["action_items"]   = _dedup_by_key(merged["action_items"],   "task")
    merged["decisions"]      = _dedup_by_key(merged["decisions"],      "decision")
    merged["blockers"]       = _dedup_by_key(merged["blockers"],       "blocker")
    merged["risks"]          = _dedup_by_key(merged["risks"],          "risk")
    merged["open_questions"] = _dedup_by_key(merged["open_questions"], "question")

    seen: set[str] = set()
    unique_topics: list[str] = []
    for t in merged["mentioned_topics"]:
        n = _normalize(t)
        if n not in seen:
            seen.add(n)
            unique_topics.append(t)
    merged["mentioned_topics"] = unique_topics
    return merged


# ─── EVIDENCE CHECK ───────────────────────────────────────────────────────────

def evidence_check(merged: dict) -> dict:
    """Downgrade confidence to 'low' for items with neither evidence nor source_time."""
    for key in ("action_items", "decisions"):
        for item in merged.get(key, []):
            if not item.get("evidence") and not item.get("source_time"):
                if item.get("confidence") in ("high", "medium"):
                    item["confidence"] = "low"
    return merged


# ─── REDUCE: final summary ────────────────────────────────────────────────────

_REDUCE_SYSTEM = """\
Ты — аналитик рабочих встреч. Тебе передан структурированный анализ планёрки.
Сформируй итоговую заметку в формате **Obsidian-ready Markdown** строго по шаблону ниже.

ЖЁСТКИЕ ТРЕБОВАНИЯ К ВЫВОДУ:
1. Возвращай ТОЛЬКО Markdown. Никакого текста до или после, никаких ```markdown ограждений вокруг всего ответа.
2. НЕ используй HTML-теги вообще (никаких <br>, <div>, <details>, <summary>, &nbsp; и пр.).
3. Action items пиши в формате **Obsidian Tasks**:
   - [ ] Кратко суть задачи 👤 @Имя 📅 YYYY-MM-DD
   Если ответственный неизвестен — опусти "👤 @...". Если дедлайн неизвестен — опусти "📅 ...".
4. Ключевые сущности (продукты, технологии, проекты, команды, важные термины) оборачивай в wiki-links: [[Docker]], [[Whisper]], [[Obsidian]], [[Backend]], [[API]] и т.п. Названия разделов системы тоже оборачивай: [[Ценообразование]], [[Клиенты]], [[Панель сборщика]]. НЕ оборачивай людей.
5. Обязательно добавь раздел "## Карта обсуждения" с Mermaid mindmap внутри ```mermaid``` блока.
6. Сохраняй язык оригинала транскрипта (если транскрипт на русском — пиши по-русски).
7. Не выдумывай факты — используй только переданные данные. Если поля нет — пиши "не указан".
8. ОБЯЗАТЕЛЬНО используй данные из section_statuses, sequence, business_context. Это ключевая операционная информация — не сворачивай её в общее «обсудили дизайн».
9. Action items должны быть КОНКРЕТНЫМИ — не «правки в панели сборщика», а «добавить инвентаризацию в [[Панель сборщика]]».

ШАБЛОН (соблюдай порядок и точные заголовки):

---
type: meeting-summary
---

## Кратко
2–5 предложений: о чём была встреча и главный итог. Упомяни главные приоритеты и текущий этап работ.

## Карта обсуждения
```mermaid
mindmap
  root((Планёрка))
    Тема1
      Подтема1
      Подтема2
    Тема2
      Подтема1
    Решения
      Решение1
    Риски
      Риск1
```
(Заполни реальными темами/разделами из анализа. Минимум 3 ветки, максимум 8. Короткие 1-3-словные узлы. Без кавычек внутри узлов.)

## Статус по разделам
| Раздел | Статус | Приоритет | Почему | Что делать |
|---|---|---|---|---|
(Заполни из section_statuses. Статус словами: «есть», «нет», «в разработке», «требует переделки», «отложено», «убрано». Приоритет: «высокий», «средний», «низкий», «—». Колонка «Почему» — из reasoning. Колонка «Что делать» — из notes. Если разделов нет — таблицу всё равно оставь с шапкой.)

## Последовательность работ
Пронумерованный список из sequence (по полю `order`), отражающий порядок и зависимости. Формат:
1. **Сейчас в работе:** … (что и от кого зависит)
2. **Дальше:** …
3. **Не приоритет / отложено:** …
Если данных нет — напиши «Чёткая последовательность не зафиксирована».

## Бизнес-контекст
Маркированный список из business_context. Каждый пункт: **тема** — почему важно — детали.
Если данных нет — раздел опусти.

## Действия (Action Items)
- [ ] Описание задачи 👤 @Имя 📅 2025-01-15
- [ ] Другая задача 👤 @Имя
(Все action_items сюда, конкретными формулировками. Группируй по разделам если их много.)

## Принятые решения
| Решение | Контекст | Участники | Подтверждение |
|---|---|---|---|

## Блокеры
| Блокер | Влияние | Ответственный | Источник | Подтверждение |
|---|---|---|---|---|
(В колонке «Источник» пиши «явный» если inferred=false, «выведено из контекста» если inferred=true.)

## Риски
| Риск | Severity | Контекст | Источник | Подтверждение |
|---|---|---|---|---|

## Открытые вопросы
| Вопрос | Ответственный | Контекст | Подтверждение |
|---|---|---|---|

ПРАВИЛА:
- Если ответственный неизвестен — пиши "не указан".
- Если дедлайн неизвестен — опусти 📅 (или пиши "не указан" в таблице).
- Для каждого пункта в таблицах добавляй evidence/source_time если есть.
- Если раздел пустой — оставь только заголовок и пустую таблицу с шапкой.\
"""


def _strip_leading_frontmatter(md: str) -> str:
    """Drop the first YAML frontmatter block if the document starts with it."""
    s = md.lstrip("\ufeff")
    if not s.startswith("---"):
        return md
    # Find closing '---' on its own line
    m = re.search(r"^---\s*\n(.*?)\n---\s*\n", s, flags=re.DOTALL)
    if not m:
        return md
    return s[m.end():]


def generate_final_summary(client: "OpenAI", model: str, merged: dict) -> str:
    input_data = {
        "chunk_summaries":   [cs["summary"] for cs in merged.get("chunk_summaries", [])],
        "section_statuses":  merged.get("section_statuses", []),
        "sequence":          merged.get("sequence", []),
        "business_context":  merged.get("business_context", []),
        "decisions":         merged.get("decisions", []),
        "action_items":      merged.get("action_items", []),
        "blockers":          merged.get("blockers", []),
        "risks":              merged.get("risks", []),
        "open_questions":    merged.get("open_questions", []),
        "important_facts":   merged.get("important_facts", []),
        "mentioned_topics":  merged.get("mentioned_topics", []),
    }
    return _chat(
        client, model, _REDUCE_SYSTEM,
        f"Данные анализа встречи:\n\n{json.dumps(input_data, ensure_ascii=False, indent=2)}",
    )


_TITLE_SYSTEM = (
    "Ты помогаешь именовать рабочие совещания. "
    "По данным встречи придумай короткий заголовок (4–8 слов) на русском, "
    "отражающий главную тему. Без кавычек, без точки в конце, "
    "без префиксов вроде 'Совещание:' или 'Планёрка:'. "
    "Верни ТОЛЬКО заголовок одной строкой."
)


def generate_meeting_title(client: "OpenAI", model: str, merged: dict, summary_md: str) -> str:
    topics    = merged.get("mentioned_topics", [])[:8]
    decisions = [d.get("text", "") if isinstance(d, dict) else str(d)
                 for d in merged.get("decisions", [])[:5]]
    actions   = [a.get("text", "") if isinstance(a, dict) else str(a)
                 for a in merged.get("action_items", [])[:5]]
    payload = {
        "topics":    topics,
        "decisions": decisions,
        "actions":   actions,
        "summary_preview": summary_md[:1500],
    }
    try:
        raw = _chat(
            client, model, _TITLE_SYSTEM,
            f"Данные встречи:\n\n{json.dumps(payload, ensure_ascii=False, indent=2)}",
        )
        title = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()
        title = title.splitlines()[0].strip().strip('"').strip("'").rstrip(".")
        # Sanity-clip
        if len(title) > 120:
            title = title[:120].rstrip() + "…"
        return title or "Совещание"
    except Exception as e:
        print(f"[WARN] Title generation failed: {e}", file=sys.stderr)
        return "Совещание"


# ─── Main pipeline ────────────────────────────────────────────────────────────

def summarize(speakers_file: Path) -> None:
    output_dir = speakers_file.parent
    stem = speakers_file.name.replace("_speakers.txt", "")

    # Read transcript
    transcript = speakers_file.read_text(encoding="utf-8")
    if not transcript.strip():
        print("[WARN] Empty transcript, skipping.", file=sys.stderr)
        return

    # Apply speaker_map.json
    speaker_map_path = output_dir / "speaker_map.json"
    if speaker_map_path.exists():
        try:
            speaker_map = json.loads(speaker_map_path.read_text(encoding="utf-8"))
            transcript = _apply_speaker_map(transcript, speaker_map)
            print(f"[INFO] speaker_map applied ({len(speaker_map)} speakers)")
        except Exception as e:
            print(f"[WARN] speaker_map.json error: {e}", file=sys.stderr)

    # Parse speaker blocks
    blocks = parse_speaker_blocks(transcript)
    if not blocks:
        print("[WARN] No speaker blocks parsed — falling back to raw text.", file=sys.stderr)
        blocks = [SpeakerBlock("SPEAKER", "00:00:00", "00:00:00", transcript)]

    print(f"[INFO] Speaker blocks: {len(blocks)}")

    # Build chunks with overlap
    chunks = build_chunks_with_overlap(blocks, CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS)
    print(f"[INFO] Chunks: {len(chunks)} "
          f"(size≤{CHUNK_SIZE_CHARS} chars, overlap≈{CHUNK_OVERLAP_CHARS} chars)")

    # LLM client
    client, model = _get_llm_client()
    print(f"[INFO] LLM: {model}")

    # ── Anonymizer setup ──────────────────────────────────────────────────────
    anonymizer = None
    if os.environ.get("ENABLE_ANONYMIZATION", "false").lower() in ("true", "1", "yes"):
        if _ANON_AVAILABLE:
            map_path = output_dir / f"{stem}_anonymization_map.json"
            language = os.environ.get("LANGUAGE", "ru")
            anonymizer = _Anonymizer(map_path=map_path, language=language, enabled=True)
            print("[INFO] Anonymization: ENABLED — PII will be masked before LLM call")
        else:
            print("[WARN] ENABLE_ANONYMIZATION=true but anonymize.py not found",
                  file=sys.stderr)
    else:
        print("[INFO] Anonymization: disabled (ENABLE_ANONYMIZATION=true to enable)")

    # Intermediate dir
    inter_dir = output_dir / "intermediate"
    inter_dir.mkdir(exist_ok=True)

    # ── MAP: extract per chunk ────────────────────────────────────────────────
    chunk_results: list[dict] = []
    for i, chunk in enumerate(chunks, 1):
        total_chars = sum(b.char_len() for b in chunk)
        print(f"[INFO] MAP chunk {i}/{len(chunks)}  "
              f"[{chunk[0].start_time}–{chunk[-1].end_time}]  "
              f"{len(chunk)} blocks  {total_chars} chars")

        # Save anonymized text for inspection (ANONYMIZE_SAVE_DEBUG=true)
        if anonymizer is not None and os.environ.get("ANONYMIZE_SAVE_DEBUG", "false").lower() in ("true", "1", "yes"):
            anon_text = anonymizer.anonymize(chunk_to_text(chunk))
            anon_path = inter_dir / f"{stem}_chunk_{i:02d}_anon.txt"
            anon_path.write_text(anon_text, encoding="utf-8")

        result = extract_chunk(client, model, chunk, i, len(chunks),
                               anonymizer=anonymizer)
        chunk_results.append(result)

        inter_path = inter_dir / f"{stem}_chunk_{i:02d}.json"
        inter_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ── COMBINED ANONYMIZED TRANSCRIPT (when anon is enabled) ─────────────────
    if anonymizer is not None:
        full_text = speakers_file.read_text(encoding="utf-8")
        combined_anon = anonymizer.anonymize(full_text)
        anon_combined_path = output_dir / f"{stem}_anonymized_speakers.txt"
        anon_combined_path.write_text(combined_anon, encoding="utf-8")
        print(f"[INFO] Anonymized transcript saved: {anon_combined_path.name}")

    # ── MERGE ────────────────────────────────────────────────────────────────
    merged = merge_chunks(chunk_results)

    # ── DEDUP ─────────────────────────────────────────────────────────────────
    before = {k: len(merged[k]) for k in ("action_items", "decisions", "risks")}
    merged = deduplicate(merged)
    after  = {k: len(merged[k]) for k in ("action_items", "decisions", "risks")}
    print(f"[INFO] Dedup: actions {before['action_items']}→{after['action_items']}  "
          f"decisions {before['decisions']}→{after['decisions']}  "
          f"risks {before['risks']}→{after['risks']}")

    # ── EVIDENCE CHECK ────────────────────────────────────────────────────────
    merged = evidence_check(merged)

    # Metadata
    merged["meeting_id"]      = stem
    merged["total_chunks"]    = len(chunks)
    merged["speaker_blocks"]  = len(blocks)

    # ── Save *_analysis.json ──────────────────────────────────────────────────
    analysis_path = output_dir / f"{stem}_analysis.json"
    analysis_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK]   {analysis_path.name}")

    # ── Save *_actions.json ───────────────────────────────────────────────────
    actions = [{**a, "status": "open"} for a in merged.get("action_items", [])]
    actions_path = output_dir / f"{stem}_actions.json"
    actions_path.write_text(
        json.dumps(actions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK]   {actions_path.name}")

    # ── Save *_decisions.json ─────────────────────────────────────────────────
    decisions_path = output_dir / f"{stem}_decisions.json"
    decisions_path.write_text(
        json.dumps(merged.get("decisions", []), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK]   {decisions_path.name}")

    # ── REDUCE: final summary ─────────────────────────────────────────────────
    print("[INFO] Generating final summary (REDUCE)...")
    summary_md = generate_final_summary(client, model, merged)
    # Strip potential thinking blocks from summary too
    summary_md = re.sub(r"<think>[\s\S]*?</think>", "", summary_md).strip()

    # ── TITLE: ask LLM for a short meeting title ──────────────────────────────
    print("[INFO] Generating meeting title...")
    title = generate_meeting_title(client, model, merged, summary_md)
    print(f"[OK]   Title: {title}")

    # ── Collect attendees (unique speakers) ───────────────────────────────────
    attendees = sorted({b.speaker for b in blocks if b.speaker})

    # ── Build Obsidian-friendly frontmatter ───────────────────────────────────
    from datetime import date as _date
    meeting_date = _date.today().isoformat()
    category     = output_dir.name

    def _yaml_list(items: list[str]) -> str:
        if not items:
            return "[]"
        return "[" + ", ".join(json.dumps(s, ensure_ascii=False) for s in items) + "]"

    frontmatter = (
        "---\n"
        f"title: {json.dumps(title, ensure_ascii=False)}\n"
        f"date: {meeting_date}\n"
        f"category: {json.dumps(category, ensure_ascii=False)}\n"
        f"attendees: {_yaml_list(attendees)}\n"
        f"tags: [meeting, {category.lower()}]\n"
        f"source: \"[[{stem}_speakers]]\"\n"
        "---\n\n"
        f"# {title}\n\n"
    )

    summary_md_full = frontmatter + _strip_leading_frontmatter(summary_md)

    summary_path = output_dir / f"{stem}_summary.md"
    summary_path.write_text(summary_md_full, encoding="utf-8")
    print(f"[OK]   {summary_path.name}")

    # ── Save meta.json for the web UI / archive ───────────────────────────────
    meta = {
        "title":     title,
        "date":      meeting_date,
        "category":  category,
        "attendees": attendees,
        "stem":      stem,
    }
    meta_path = output_dir / f"{stem}_meta.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK]   {meta_path.name}")

    print(f"[OK] Pipeline complete → {output_dir}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Meeting Intelligence Pipeline — evidence-first structured extraction"
    )
    parser.add_argument(
        "--speakers-file", required=True,
        help="Path to *_speakers.txt (output of postprocess.py)",
    )
    args = parser.parse_args()

    path = Path(args.speakers_file)
    if not path.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)

    summarize(path)


if __name__ == "__main__":
    main()
