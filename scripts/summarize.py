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

# Summary output verbosity. Influences both REDUCE prompt template and what we
# feed to the model (less input + shorter output = faster generation locally).
#   full     — все разделы (как раньше)
#   compact  — TL;DR, коуч, решения (top), действия (фильтр), вопросы, риски+блокеры
#   minimal  — только TL;DR + коуч + действия
SUMMARY_MODE = os.environ.get("SUMMARY_MODE", "compact").strip().lower()
if SUMMARY_MODE not in ("full", "compact", "minimal"):
    SUMMARY_MODE = "compact"

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

    # Локальные модели на CPU/малом GPU могут думать > 10 мин на чанк.
    # Поэтому даём большой таймаут и НЕ ретраим (иначе будет 3×timeout).
    timeout_s = float(os.environ.get("LLM_TIMEOUT_SECONDS", "3600"))  # 1 час
    max_retries = int(os.environ.get("LLM_MAX_RETRIES", "0"))
    print(f"[INFO] LLM client: timeout={timeout_s:.0f}s, max_retries={max_retries}")

    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout_s,
        max_retries=max_retries,
    ), model


def _chat(client: "OpenAI", model: str, system: str, user: str) -> str:
    kwargs: dict = dict(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.1,
    )
    # Ollama native "think" flag (прокидывается через openai-compat extra_body).
    # Безопасно для других провайдеров — неизвестные поля игнорируются.
    if _LOG_THINKING:
        kwargs["extra_body"] = {"think": True}
        kwargs["stream"] = True
        return _chat_streaming(client, kwargs)

    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    reasoning = (getattr(msg, "reasoning_content", None)
                 or getattr(msg, "reasoning", None)
                 or getattr(msg, "thinking", None)
                 or "") or ""
    if _LOG_THINKING:
        if reasoning:
            _emit_thinking(reasoning, source="reasoning_content")
        elif "<think>" in content.lower():
            pass  # будет извлечено в _clean_llm_response → _log_thinking
        else:
            print("[THINK] (пусто: модель не вернула ни <think>…</think>, "
                  "ни reasoning_content). Проверь, что в Ollama включён thinking "
                  "для qwen3 и в промпте нет /no_think.", file=sys.stderr, flush=True)
    return content


def _chat_streaming(client: "OpenAI", kwargs: dict) -> str:
    """Stream thinking tokens to stderr live; collect final content for return."""
    print("[THINK ▶ stream] ──────────────────────────────", file=sys.stderr, flush=True)
    content_parts: list[str] = []
    think_buf = ""           # буфер по строкам — печатаем целыми строками
    in_think_tag = False     # для случая, если размышления приходят в <think> внутри content
    saw_any_thinking = False

    try:
        stream = client.chat.completions.create(**kwargs)
        for event in stream:
            try:
                delta = event.choices[0].delta
            except (IndexError, AttributeError):
                continue

            # 1) Отдельное поле reasoning_content / reasoning / thinking в delta
            r = (getattr(delta, "reasoning_content", None)
                 or getattr(delta, "reasoning", None)
                 or getattr(delta, "thinking", None))
            if r:
                saw_any_thinking = True
                think_buf += r
                while "\n" in think_buf:
                    line, think_buf = think_buf.split("\n", 1)
                    print(f"[THINK] {line}", file=sys.stderr, flush=True)

            # 2) Обычный content (может содержать <think>…</think> у некоторых рантаймов)
            c = getattr(delta, "content", None)
            if c:
                if "<think>" in c.lower() or in_think_tag:
                    saw_any_thinking = True
                    # очень упрощённый стейт-машина для тегов
                    chunk = c
                    while chunk:
                        if not in_think_tag:
                            i = chunk.lower().find("<think>")
                            if i < 0:
                                content_parts.append(chunk)
                                break
                            content_parts.append(chunk[:i])
                            chunk = chunk[i + len("<think>"):]
                            in_think_tag = True
                        else:
                            j = chunk.lower().find("</think>")
                            if j < 0:
                                think_buf += chunk
                                chunk = ""
                            else:
                                think_buf += chunk[:j]
                                chunk = chunk[j + len("</think>"):]
                                in_think_tag = False
                            while "\n" in think_buf:
                                line, think_buf = think_buf.split("\n", 1)
                                print(f"[THINK] {line}", file=sys.stderr, flush=True)
                else:
                    content_parts.append(c)
    finally:
        if think_buf.strip():
            print(f"[THINK] {think_buf.rstrip()}", file=sys.stderr, flush=True)
        if not saw_any_thinking:
            print("[THINK] (пусто: thinking-токены не пришли в стриме)",
                  file=sys.stderr, flush=True)
        print("[THINK ◀ stream] ── end ───────────────────────",
              file=sys.stderr, flush=True)

    return "".join(content_parts).strip()


_LOG_THINKING = os.environ.get("LLM_LOG_THINKING", "0").lower() in ("1", "true", "yes", "on")
_THINK_RE = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)


def _emit_thinking(text: str, source: str = "think-tag") -> None:
    text = (text or "").strip()
    if not text:
        return
    print(f"[THINK ▶ {source}] ──────────────────────────────", file=sys.stderr, flush=True)
    for line in text.splitlines():
        print(f"[THINK] {line}", file=sys.stderr, flush=True)
    print(f"[THINK ◀ {source}] ── end ───────────────────────", file=sys.stderr, flush=True)


def _log_thinking(raw: str) -> None:
    """If LLM_LOG_THINKING is on, print <think>…</think> blocks to stderr."""
    if not _LOG_THINKING:
        return
    for i, m in enumerate(_THINK_RE.finditer(raw), 1):
        _emit_thinking(m.group(1), source=f"think-tag #{i}")


def _clean_llm_response(raw: str) -> str:
    """Strip <think>…</think> blocks (Qwen3 thinking mode) and markdown fences."""
    _log_thinking(raw)
    raw = _THINK_RE.sub("", raw).strip()
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

  "meeting_type": "planning|audit|review|kickoff|brainstorm|standup|one_on_one|retro|demo|other",
  "meeting_context": "1-3 предложения: фон встречи — что было ДО встречи, на каком этапе проект, кто подрядчики/участники, какая история. Не путать с пересказом — это именно исторический контекст (напр. 'подрядчик сдал работу, качество спорное'). Пиши только если в фрагменте явно звучит; иначе null.",

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
      "quote": "прямая цитата, фиксирующая решение (до ш2 слов), обязательно из реального текста или null",
      "source_time": "HH:MM:SS или null",
      "evidence": "короткая цитата из transcript (может совпадать с quote)",
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
      "positions": [
        {"participant": "имя", "stance": "позиция этого человека кратко"}
      ],
      "source_time": "HH:MM:SS или null"
    }
  ],

  "meta_observations": [
    {
      "observation": "наблюдение О ПРОЦЕССЕ встречи, не о содержании. Примеры: 'встреча без повестки', 'фиксация решений держится на одном человеке', 'нет процесса приёмки', 'приоритеты выясняются в конце', 'возвраты к пройденному', 'задачи без владельцев/сроков'",
      "severity": "high|medium|low",
      "root_cause": "корневая причина или null",
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
- meeting_type: определи ОДИН раз по характеру фрагмента (в merge мы возьмём наиболее частый). audit — если разбирают сданную работу; planning — если планируют; review — если смотрят результаты.
- meeting_context: ОЧЕНЬ важно. Это ИСТОРИЯ проекта, которую люди упоминают между прочим («это нам подрядчик сделал», «мы уже заплатили», «работаем в таблицах пока»). Это НЕ пересказ встречи.
- meta_observations — это наблюдения ОБ ОРГАНИЗАЦИИ встречи (не было повестки, всё держится на одном человеке, нет владельцев у задач, нет процесса приёмки работ и т.п.). Не путать с бизнес-рисками. Пиши только если явно видно в фрагменте.
- decisions[].quote — только реальные слова спикера из transcript, без перефраза. Отрежь до самой сути.
- open_questions[].positions — включай только если в обсуждении были разные мнения участников; иначе пустой массив.
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
            "meeting_type": None, "meeting_context": None,
            "section_statuses": [], "sequence": [], "business_context": [],
            "decisions": [], "action_items": [], "blockers": [],
            "risks": [], "open_questions": [], "meta_observations": [],
            "important_facts": [],
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
        "meeting_type_votes": [],
        "meeting_context_parts": [],
        "section_statuses":  [],
        "sequence":          [],
        "business_context":  [],
        "decisions":         [],
        "action_items":      [],
        "blockers":          [],
        "risks":              [],
        "open_questions":    [],
        "meta_observations": [],
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
        mt = cr.get("meeting_type")
        if mt and isinstance(mt, str):
            merged["meeting_type_votes"].append(mt.strip().lower())
        mc = cr.get("meeting_context")
        if mc and isinstance(mc, str) and mc.strip():
            merged["meeting_context_parts"].append(mc.strip())
        for key in ("section_statuses", "sequence", "business_context",
                    "decisions", "action_items", "blockers", "risks",
                    "open_questions", "meta_observations",
                    "important_facts", "mentioned_topics"):
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
    merged["meta_observations"] = _dedup_by_key(
        merged.get("meta_observations", []), "observation")

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
Ты — аналитик рабочих встреч. Тебе передан структурированный анализ.
Сформируй итоговую заметку в формате **Obsidian-ready Markdown** строго по шаблону ниже.

ЖЁСТКИЕ ТРЕБОВАНИЯ К ВЫВОДУ:
1. Возвращай ТОЛЬКО Markdown. Никакого текста до или после, никаких ```markdown ограждений вокруг всего ответа.
2. НЕ используй HTML-теги вообще (никаких <br>, <div>, <details>, <summary>, &nbsp; и пр.).
3. Action items пиши КАК ОБЫЧНЫЙ МАРКИРОВАННЫЙ СПИСОК (НЕ чекбоксы, НЕ Obsidian Tasks). Формат:
   - **Суть задачи** — ответственный: Имя, срок: YYYY-MM-DD.
   Если ответственного нет — пиши «ответственный: не назначен». Если срока нет — «срок: не назначен». Никаких эмодзи (👤📅), никаких `[ ]`.
4. Ключевые сущности (продукты, технологии, проекты, разделы системы) оборачивай в wiki-links: [[Docker]], [[Ценообразование]], [[Панель сборщика]], [[Telegram]], [[Backend]]. НЕ оборачивай людей.
5. Обязательно добавь раздел "## Карта обсуждения" с Mermaid mindmap внутри ```mermaid``` блока.
6. Сохраняй язык оригинала транскрипта (если транскрипт на русском — пиши по-русски).
7. Не выдумывай факты. Используй только переданные данные. Если поля нет — пиши "не указан" или опускай блок (по правилам ниже).
8. ОБЯЗАТЕЛЬНО используй данные из section_statuses, sequence, business_context, meta_observations.
9. Action items должны быть КОНКРЕТНЫМИ — не «правки в панели сборщика», а «добавить инвентаризацию в [[Панель сборщика]]».
10. В блоках «Принятые решения» цитата (поле `quote`) выводится курсивом и с тайм-кодом.
11. В «Открытых вопросах» если есть `positions` — выведи позиции сторон в скобках «(Имя — позиция; Имя — позиция)».
12. Если meta.orphan_pct ≥ 50 ИЛИ нет ни одного action item с owner+deadline — обязательно добавь в «Наблюдения о встрече» пункт «Активность ≠ решения: задачи зафиксированы без владельцев/сроков».
13. Если meta.anonymizer_artifacts_present = true — обязательно добавь блок «## Оговорка по источнику» в самом конце.
14. Блок «## Взгляд бизнес-коуча» — это ТВОЙ собственный аналитический разбор поверх данных. Не цитируй сырые поля, а синтезируй: что главное, в каком порядке делать, какие риски недооценены, как улучшить следующую встречу. Пиши конкретно, без воды, без общих фраз про «вовлечённость» и «синергию».

ШАБЛОН (соблюдай порядок и точные заголовки):

---
type: meeting-summary
meeting_type: <meta.meeting_type_key или "other">
---

> [!info] Метаданные
> **Тип:** <meta.meeting_type_label>
> **Длительность:** <от meta.time_start до meta.time_end, если оба есть; иначе «не указана»>
> **Участники:** <через запятую из meta.participants; если пусто — «не указаны»>

## Кратко
2–5 предложений нарративом: о чём была встреча, главный итог, ключевые приоритеты, текущий этап работ. Стиль: связный текст, не bullet'ы.

## Контекст
1–3 предложения: фон встречи (что было ДО встречи, на каком этапе проект, кто подрядчики, какая история).
Если meta.meeting_context = null И business_context пуст — опусти этот блок целиком.

## Действия (Action Items)
- **Описание задачи** — ответственный: Имя, срок: 2025-01-15.
- **Другая задача** — ответственный: Имя, срок: не назначен.

Если действий много (>5) — сгруппируй по разделу (поле `section`):
### <Название раздела>
- **<Задача>** — ответственный: …, срок: …

Если задач нет — «Конкретных action items не зафиксировано».

## Взгляд бизнес-коуча
> [!tip] Аналитика и рекомендации
> Это синтез поверх данных. Без воды.

**Главные приоритеты (top-3):**
1. <приоритет 1 — почему именно он, что заблокирует если не сделать>
2. <приоритет 2>
3. <приоритет 3>

**Предлагаемая стратегия:**
2–4 предложения: в каком порядке двигаться, что распараллелить, что отложить. Опирайся на section_statuses (статус «нет» с высоким приоритетом — кандидат на первый ход) и sequence (зависимости).

**Слепые зоны и недооценённые риски:**
- <риск/блокер, который команда явно НЕ обсудила или преуменьшила — выводи из inferred=true блокеров/рисков и отсутствия процессов в meta_observations>
- <…>

**Как провести следующую встречу продуктивнее:**
- <конкретный совет, привязанный к meta_observations или к тому что видно по данным: повестка, тайминг, фиксация владельцев, критерии приёмки, и т.д.>
- <…>

(Этот блок пиши ВСЕГДА, даже если данных мало — тогда отметь это явно: «Данных мало, гипотетически:…». Минимум 3 приоритета, минимум 2 совета по встрече.)

## Карта обсуждения
```mermaid
mindmap
  root((Встреча))
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
(Заполни реальными темами/разделами. Минимум 3 ветки, максимум 8. Короткие 1–3-словные узлы. Без кавычек.)

## Статус по разделам
| Раздел | Статус | Приоритет | Почему | Что делать |
|---|---|---|---|---|
(Заполни из section_statuses. Статус словами: «есть», «нет», «в разработке», «требует переделки», «отложено», «убрано». Приоритет: «высокий», «средний», «низкий», «—». Если разделов нет — таблицу с шапкой оставь.)

## Последовательность работ
Пронумерованный список из sequence, отражающий порядок и зависимости. Группируй:
1. **Сейчас в работе:** … (что и от кого зависит)
2. **Дальше:** …
3. **Не приоритет / отложено:** …
Если данных нет — «Чёткая последовательность не зафиксирована».

## Бизнес-контекст
Маркированный список из business_context. Каждый пункт: **тема** — почему важно — детали.
Если business_context пуст — опусти раздел.

## Принятые решения
Для каждого decision выводи блок (НЕ таблица):

**1. <decision>**
> *«<quote>» (<source_time>)*
- Контекст: <context>
- Участники: <speakers через запятую или «не указаны»>

(Если quote отсутствует — опусти строку с цитатой. Если source_time нет — пиши «время не указано».)
Если решений нет — «Финальных решений не зафиксировано».

## Блокеры
| Блокер | Влияние | Ответственный | Источник | Подтверждение |
|---|---|---|---|---|
(В колонке «Источник» пиши «явный» если inferred=false, «выведено из контекста» если inferred=true.)

## Риски
| Риск | Severity | Контекст | Источник | Подтверждение |
|---|---|---|---|---|

## Открытые вопросы
Для каждого вопроса — отдельный пункт:
- **<question>** — <context>. Ответственный: <owner или «не указан»>.
  - Позиции: <если positions есть — «Имя — позиция; Имя — позиция»; иначе строку опусти>

## Наблюдения о встрече
Маркированный список из meta_observations. Формат:
- **<observation>** [severity: <severity>]. <root_cause если есть>.

Также сюда добавляй системные наблюдения:
- Если meta.orphan_pct ≥ 50: «**Активность ≠ решения** [severity: средняя]. Из <total> задач <orphaned> без владельца или срока. Список дел сам по себе не равен прогрессу.»

Если ни meta_observations, ни системных наблюдений нет — раздел опусти.

## Оговорка по источнику
(Включай только если meta.anonymizer_artifacts_present = true.)
Транскрипт прошёл анонимизацию: имена, компании и контакты заменены тегами вида PERSON_NNN, COMPANY_NNN. Часть тайм-кодов могла быть искажена анонимайзером — относись к ним как к приблизительным. Имена в этой заметке восстановлены через обратное сопоставление; если встретился тег вместо имени — значит сопоставление не сработало для конкретной сущности.

ПРАВИЛА:
- Если ответственный неизвестен — пиши «не назначен».
- Если дедлайн неизвестен — пиши «не назначен».
- Не дублируй информацию между блоками: контекст встречи в «Контекст», бизнес-обоснование в «Бизнес-контекст», процессные наблюдения в «Наблюдения о встрече», синтез и стратегия — только во «Взгляд бизнес-коуча».\
"""


# ─── REDUCE: COMPACT mode ─────────────────────────────────────────────────────

_REDUCE_SYSTEM_COMPACT = """\
Ты — аналитик рабочих встреч. Тебе передан структурированный анализ.
Сформируй КОРОТКУЮ итоговую заметку в формате **Obsidian-ready Markdown** строго по шаблону ниже.

ЦЕЛЬ: минимум воды, максимум смысла. Читатель должен за 1–2 минуты понять главное.

ЖЁСТКИЕ ТРЕБОВАНИЯ:
1. Возвращай ТОЛЬКО Markdown. Без HTML-тегов, без ``` вокруг всего ответа.
2. Action items: обычный маркированный список:
   - **Суть задачи** — ответственный: Имя, срок: YYYY-MM-DD.
   Если ответственного нет — «не назначен», срока нет — «не назначен». Никаких чекбоксов.
3. Ключевые сущности (продукты, технологии, проекты, разделы) оборачивай в [[wiki-links]]. Людей НЕ оборачивай.
4. Сохраняй язык транскрипта.
5. Не выдумывай факты. Используй только переданные данные.
6. Раздел «Взгляд бизнес-коуча» — это ТВОЙ синтез. Не цитируй сырые поля, синтезируй.

ШАБЛОН (соблюдай порядок и заголовки):

---
type: meeting-summary
meeting_type: <meta.meeting_type_key или "other">
---

> [!info] Метаданные
> **Тип:** <meta.meeting_type_label>
> **Длительность:** <meta.time_start–meta.time_end или «не указана»>
> **Участники:** <meta.participants через запятую или «не указаны»>

## TL;DR
3–5 предложений нарративом: о чём встреча, главный итог, ключевые приоритеты, текущий этап. Это самостоятельная выжимка — кто прочитал только её, должен понять суть.

## Взгляд бизнес-коуча
> [!tip] Аналитика и рекомендации
> Синтез поверх данных. Без воды.

**Главные приоритеты (top-3):**
1. <приоритет 1 — почему именно он, что заблокирует если не сделать>
2. <приоритет 2>
3. <приоритет 3>

**Предлагаемая стратегия:**
2–4 предложения: порядок действий, что распараллелить, что отложить. Опирайся на section_statuses (статус «нет» с высоким приоритетом — первый ход) и sequence (зависимости).

**Слепые зоны и недооценённые риски:**
- <риск/блокер, который команда явно НЕ обсудила или преуменьшила — выводи из inferred=true в risks/blockers и из пробелов в meta_observations>
- <…> (2–4 пункта)

**Главные процессные проблемы встречи:**
- <из meta_observations — только самые важные, 2–3 пункта. Если meta.orphan_pct ≥ 50 — обязательно: «Из <total> задач <orphaned> без владельца/срока — активность ≠ решения».>

**Как провести следующую встречу продуктивнее:**
- <конкретный совет 1>
- <конкретный совет 2>
- <конкретный совет 3>

## Ключевые решения
Максимум 5–7 самых значимых решений. Формат:

**1. <decision>**
> *«<quote>» (<source_time>)*
- Контекст: <context кратко>
- Участники: <speakers или «не указаны»>

Если решений нет — «Финальных решений не зафиксировано».

## Действия
Только задачи с владельцем ИЛИ сроком ИЛИ привязкой к разделу. Если таких нет — выведи топ-5 самых конкретных. Группируй по разделу, если задач >5:

### <Раздел>
- **<Задача>** — ответственный: …, срок: …

Если осталось много «сирот» (без владельца, срока и раздела) — добавь в конце одним абзацем: «Кроме того, без владельцев и сроков зафиксированы: <через запятую краткие формулировки>».

Если задач нет — «Конкретных action items не зафиксировано».

## Открытые вопросы
Топ-5 нерешённых вопросов. Каждый — один пункт:
- **<question>** — <context в одном предложении>. <если есть позиции — «Позиции: Имя — …; Имя — …»>

Если вопросов нет — раздел опусти.

## Риски и блокеры
Объединённая таблица. Максимум 7 строк. Сортируй: блокеры с явным impact → high-риски → medium-риски.

| Тип | Описание | Severity / Impact | Источник |
|---|---|---|---|
(Тип = «блокер» или «риск». Источник = «явный» если inferred=false, «контекст» если inferred=true.)

Если ни рисков ни блокеров — раздел опусти.

## Оговорка по источнику
(Включай только если meta.anonymizer_artifacts_present = true.)
Транскрипт прошёл анонимизацию: имена, компании и контакты заменены тегами PERSON_NNN, COMPANY_NNN и т.п. Имена в этой заметке восстановлены через обратное сопоставление; если встретился тег — значит сопоставление не сработало для конкретной сущности.

ПРАВИЛА:
- Без раздела «Карта обсуждения», «Статус по разделам», «Последовательность работ», «Бизнес-контекст», «Контекст», «Наблюдения о встрече» — их содержание уходит во «Взгляд бизнес-коуча».
- Без HTML, без эмодзи, без чекбоксов.\
"""


# ─── REDUCE: MINIMAL mode ─────────────────────────────────────────────────────

_REDUCE_SYSTEM_MINIMAL = """\
Ты — аналитик встреч. Сделай ОЧЕНЬ короткую заметку в Markdown.

Возвращай ТОЛЬКО Markdown, без HTML, без ``` вокруг ответа. Сохраняй язык транскрипта.
Ключевые сущности (продукты, технологии, разделы) — в [[wiki-links]]. Людей не оборачивай.

ШАБЛОН:

---
type: meeting-summary
meeting_type: <meta.meeting_type_key или "other">
---

> [!info] Метаданные
> **Тип:** <meta.meeting_type_label>
> **Длительность:** <meta.time_start–meta.time_end или «не указана»>
> **Участники:** <meta.participants или «не указаны»>

## TL;DR
3–5 предложений: о чём встреча, итог, приоритеты.

## Главное от коуча
**Топ-3 приоритета:**
1. <…>
2. <…>
3. <…>

**Главные риски/блокеры (≤3):**
- <…>

**Как улучшить следующую встречу:**
- <…> (2–3 пункта)

## Действия
Только с владельцем или сроком. Формат:
- **<Задача>** — ответственный: Имя, срок: YYYY-MM-DD.

Если задач нет — «Конкретных action items не зафиксировано».\
"""


def _trim_for_compact(merged: dict) -> dict:
    """Trim merged analysis to keep REDUCE prompt small (compact/minimal modes)."""
    PRI = {"high": 0, "medium": 1, "low": 2, "none": 3, None: 3}
    SEV = {"high": 0, "medium": 1, "low": 2, None: 3}

    # Decisions: prefer those with quote + source_time + high confidence
    decisions = sorted(
        merged.get("decisions", []),
        key=lambda d: (
            0 if d.get("quote") else 1,
            0 if d.get("source_time") else 1,
            {"high": 0, "medium": 1, "low": 2}.get(d.get("confidence"), 3),
        ),
    )[:7]

    # Risks/blockers: prefer explicit (inferred=false) + high severity
    risks = sorted(
        merged.get("risks", []),
        key=lambda r: (
            0 if not r.get("inferred") else 1,
            SEV.get(r.get("severity"), 3),
        ),
    )[:7]
    blockers = sorted(
        merged.get("blockers", []),
        key=lambda b: (0 if not b.get("inferred") else 1,),
    )[:5]

    # Open questions: keep top 5
    questions = merged.get("open_questions", [])[:5]

    # Meta observations: top 3 by severity
    observations = sorted(
        merged.get("meta_observations", []),
        key=lambda o: SEV.get(o.get("severity"), 3),
    )[:3]

    # Section statuses: keep only high/medium priority (для контекста коучу)
    statuses = [s for s in merged.get("section_statuses", [])
                if (s.get("priority") in ("high", "medium"))]
    statuses = sorted(statuses, key=lambda s: PRI.get(s.get("priority"), 3))[:10]

    # Sequence: keep up to 8 (для коуча, чтобы понять зависимости)
    sequence = merged.get("sequence", [])[:8]

    out = dict(merged)
    out["decisions"]         = decisions
    out["risks"]             = risks
    out["blockers"]          = blockers
    out["open_questions"]    = questions
    out["meta_observations"] = observations
    out["section_statuses"]  = statuses
    out["sequence"]          = sequence
    # business_context и important_facts в компакт-режиме не нужны
    out["business_context"]  = []
    out["important_facts"]   = []
    return out


def _trim_for_minimal(merged: dict) -> dict:
    """Minimal mode: оставляем только то, что нужно для TL;DR + коуч + действия."""
    trimmed = _trim_for_compact(merged)
    # Действия — только с owner или deadline
    def _is_concrete(a: dict) -> bool:
        o = a.get("owner"); d = a.get("deadline")
        has_o = bool(o) and str(o).strip().lower() not in ("null", "none", "")
        has_d = bool(d) and str(d).strip().lower() not in ("null", "none", "")
        return has_o or has_d
    actions = [a for a in trimmed.get("action_items", []) if _is_concrete(a)]
    if not actions:
        actions = trimmed.get("action_items", [])[:5]
    trimmed["action_items"]  = actions
    trimmed["decisions"]     = trimmed["decisions"][:3]
    trimmed["open_questions"] = []
    trimmed["risks"]         = trimmed["risks"][:3]
    trimmed["blockers"]      = trimmed["blockers"][:2]
    trimmed["section_statuses"] = []
    trimmed["sequence"]      = []
    return trimmed


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


_MEETING_TYPE_LABELS = {
    "planning":   "Планёрка",
    "audit":      "Аудит / разбор",
    "review":     "Ревью результатов",
    "kickoff":    "Kickoff",
    "brainstorm": "Брейншторм",
    "standup":    "Стендап",
    "one_on_one": "1:1",
    "retro":      "Ретро",
    "demo":       "Демо",
    "other":      "Рабочая встреча",
}


def _compute_meta_stats(merged: dict, transcript: str | None = None) -> dict:
    """Compute meeting-level meta facts to enrich the REDUCE prompt."""
    # Most-common meeting_type vote
    votes = merged.get("meeting_type_votes", [])
    mt_key = None
    if votes:
        from collections import Counter
        mt_key = Counter(votes).most_common(1)[0][0]
    mt_label = _MEETING_TYPE_LABELS.get(mt_key or "", _MEETING_TYPE_LABELS["other"])

    # Meeting context: longest non-duplicate parts joined
    ctx_parts = merged.get("meeting_context_parts", [])
    seen_ctx: list[str] = []
    for p in ctx_parts:
        if not any(_jaccard(p, s) >= 0.6 for s in seen_ctx):
            seen_ctx.append(p)
    meeting_context = " ".join(seen_ctx).strip() if seen_ctx else None

    # Action item orphan ratio (no owner OR no deadline)
    actions = merged.get("action_items", [])
    total_a = len(actions)
    orphaned = sum(
        1 for a in actions
        if not (a.get("owner") and str(a.get("owner")).lower() not in ("null", "none", ""))
        or not (a.get("deadline") and str(a.get("deadline")).lower() not in ("null", "none", ""))
    )
    orphan_pct = round(100 * orphaned / total_a) if total_a else 0

    # Anonymizer artefacts in transcript
    anon_present = False
    if transcript:
        anon_present = bool(re.search(
            r"\b(PERSON|COMPANY|LOCATION|EMAIL|PHONE|DOMAIN|URL|TRANSACTION)_\d{3,}\b",
            transcript,
        ))

    # Time range from chunk summaries
    cs = merged.get("chunk_summaries", [])
    time_start = cs[0]["time_range"].get("start") if cs and cs[0].get("time_range") else None
    time_end   = cs[-1]["time_range"].get("end")  if cs and cs[-1].get("time_range") else None

    # Participants — collected from decisions.speakers and action_items.owner
    participants: set[str] = set()
    for d in merged.get("decisions", []):
        for s in (d.get("speakers") or []):
            if s and isinstance(s, str):
                participants.add(s.strip())
    for a in merged.get("action_items", []):
        owner = a.get("owner")
        if owner and isinstance(owner, str) and owner.lower() not in ("null", "none"):
            participants.add(owner.strip())

    return {
        "meeting_type_key":   mt_key,
        "meeting_type_label": mt_label,
        "meeting_context":    meeting_context,
        "time_start":         time_start,
        "time_end":           time_end,
        "participants":       sorted(participants),
        "action_items_total": total_a,
        "action_items_orphaned": orphaned,
        "orphan_pct":         orphan_pct,
        "anonymizer_artifacts_present": anon_present,
    }


def generate_final_summary(client: "OpenAI", model: str, merged: dict,
                           transcript: str | None = None) -> str:
    meta = _compute_meta_stats(merged, transcript=transcript)

    # Pick template + trim input depending on SUMMARY_MODE
    if SUMMARY_MODE == "minimal":
        system_prompt = _REDUCE_SYSTEM_MINIMAL
        data_src = _trim_for_minimal(merged)
    elif SUMMARY_MODE == "compact":
        system_prompt = _REDUCE_SYSTEM_COMPACT
        data_src = _trim_for_compact(merged)
    else:  # full
        system_prompt = _REDUCE_SYSTEM
        data_src = merged

    print(f"[INFO] Summary mode: {SUMMARY_MODE}")

    input_data = {
        "meta":              meta,
        "chunk_summaries":   [cs["summary"] for cs in data_src.get("chunk_summaries", [])],
        "section_statuses":  data_src.get("section_statuses", []),
        "sequence":          data_src.get("sequence", []),
        "business_context":  data_src.get("business_context", []),
        "decisions":         data_src.get("decisions", []),
        "action_items":      data_src.get("action_items", []),
        "blockers":          data_src.get("blockers", []),
        "risks":              data_src.get("risks", []),
        "open_questions":    data_src.get("open_questions", []),
        "meta_observations": data_src.get("meta_observations", []),
        "important_facts":   data_src.get("important_facts", []),
        "mentioned_topics":  data_src.get("mentioned_topics", []),
    }
    return _chat(
        client, model, system_prompt,
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
    summary_md = generate_final_summary(client, model, merged, transcript=transcript)
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
