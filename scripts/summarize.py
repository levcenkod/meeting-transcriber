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
Твоя задача — извлечь структурированные факты. Не делай просто краткое резюме.

Верни ТОЛЬКО строго валидный JSON без Markdown и без пояснений. Схема:

{
  "chunk_id": <число>,
  "time_range": {"start": "HH:MM:SS", "end": "HH:MM:SS"},
  "summary": "краткое резюме фрагмента",
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
      "task": "что нужно сделать",
      "owner": "имя или null",
      "deadline": "дата или null",
      "context": "контекст поручения",
      "source_time": "HH:MM:SS или null",
      "evidence": "короткая цитата",
      "confidence": "high|medium|low"
    }
  ],
  "blockers": [
    {
      "blocker": "описание",
      "impact": "влияние",
      "owner": "имя или null",
      "source_time": "HH:MM:SS или null",
      "evidence": "короткая цитата"
    }
  ],
  "risks": [
    {
      "risk": "описание",
      "severity": "high|medium|low",
      "context": "контекст",
      "source_time": "HH:MM:SS или null",
      "evidence": "короткая цитата"
    }
  ],
  "open_questions": [
    {
      "question": "вопрос",
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
- Если owner неизвестен — owner = null.
- Если deadline неизвестен — deadline = null.
- Если уверенность низкая — confidence = "low".
- Указывай source_time если таймкод есть в тексте.
- Указывай evidence — короткую цитату из transcript.
- Если это просто обсуждение без чёткого решения — не записывай как decision.
- Если это не поручение — не записывай как action_item.
- Сохраняй имена спикеров как они есть в transcript.
- В тексте могут встречаться токены вида PERSON_001, COMPANY_001, LOCATION_001, EMAIL_001, PHONE_001, DOMAIN_001, URL_001, TRANSACTION_001 — это анонимизированные реальные сущности (имена людей, компаний, городов, стран и т.д.). Обращайся с ними как с обычными именами собственными: включай в контекст, решения, поручения и цитаты точно как они есть. Никогда не убирай, не сокращай и не заменяй эти токены.\
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
        "chunk_summaries": [],
        "decisions":       [],
        "action_items":    [],
        "blockers":        [],
        "risks":           [],
        "open_questions":  [],
        "important_facts": [],
        "mentioned_topics": [],
    }
    for cr in chunk_results:
        if cr.get("summary"):
            merged["chunk_summaries"].append({
                "chunk_id":   cr.get("chunk_id"),
                "time_range": cr.get("time_range", {}),
                "summary":    cr["summary"],
            })
        for key in ("decisions", "action_items", "blockers", "risks",
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


def deduplicate(merged: dict) -> dict:
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
Сформируй финальное Markdown-резюме строго по шаблону ниже.

# Резюме планёрки

## Кратко
(2–5 предложений: о чём была встреча и главный итог)

## Основные темы
- ...

## Принятые решения
| Решение | Контекст | Участники | Подтверждение |
|---|---|---|---|

## Поручения
| Задача | Ответственный | Дедлайн | Контекст | Подтверждение |
|---|---|---|---|---|

## Блокеры
| Блокер | Влияние | Ответственный | Подтверждение |
|---|---|---|---|

## Риски
| Риск | Severity | Контекст | Подтверждение |
|---|---|---|---|

## Открытые вопросы
| Вопрос | Ответственный | Контекст | Подтверждение |
|---|---|---|---|

Правила:
- Не выдумывай новые факты — используй только предоставленные данные.
- Если ответственный неизвестен — пиши "не указан".
- Если дедлайн неизвестен — пиши "не указан".
- Сохраняй язык оригинала транскрипта.
- Для каждого важного пункта добавляй evidence или source_time если они есть.\
"""


def generate_final_summary(client: "OpenAI", model: str, merged: dict) -> str:
    input_data = {
        "chunk_summaries":  [cs["summary"] for cs in merged.get("chunk_summaries", [])],
        "decisions":        merged.get("decisions", []),
        "action_items":     merged.get("action_items", []),
        "blockers":         merged.get("blockers", []),
        "risks":            merged.get("risks", []),
        "open_questions":   merged.get("open_questions", []),
        "important_facts":  merged.get("important_facts", []),
        "mentioned_topics": merged.get("mentioned_topics", []),
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

    summary_md_full = frontmatter + summary_md

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
