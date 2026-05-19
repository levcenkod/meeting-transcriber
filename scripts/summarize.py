#!/usr/bin/env python3
"""
summarize.py — создаёт summary.md и actions.json из *_speakers.txt через LLM.

Использование:
    python summarize.py --speakers-file /output/Логистика/meeting_speakers.txt

Результат (рядом с исходным файлом):
    meeting_summary.md
    meeting_actions.json

Переменные окружения (из .env):
    LLM_BASE_URL   — URL OpenAI-compatible API (напр. http://localhost:1234/v1)
    LLM_API_KEY    — API ключ (для LM Studio: любая строка, напр. not-required)
    LLM_MODEL      — имя модели (напр. qwen3-32b)

При запуске внутри Docker скрипт автоматически заменяет localhost →
host.docker.internal, чтобы достать LM Studio / Ollama на хост-машине.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("[ERROR] Пакет openai не установлен: pip install openai", file=sys.stderr)
    sys.exit(1)

# ─── Настройки ────────────────────────────────────────────────────────────────

CHUNK_MAX_CHARS = 12_000  # ~3 000 токенов — безопасно для большинства моделей

SYSTEM_CHUNK_SUMMARY = """\
Ты — ассистент для анализа деловых встреч.
Напиши краткое резюме предоставленного фрагмента транскрипта встречи.
Сохраняй язык оригинала. Будь конкретным и лаконичным."""

SYSTEM_FINAL_SUMMARY = """\
Ты — ассистент для анализа деловых встреч.
Тебе предоставлены резюме фрагментов одной встречи.
Создай единое финальное резюме в формате Markdown со структурой:

## Краткое резюме
(2–4 предложения)

## Основные темы
- ...

## Ключевые решения
- ...

Сохраняй язык оригинала."""

SYSTEM_ACTIONS = """\
Ты — ассистент для анализа деловых встреч.
Из предоставленных материалов извлеки все задачи и поручения (action items).
Верни ТОЛЬКО валидный JSON-массив без пояснений и без markdown-обёрток:
[
  {
    "action": "Описание задачи",
    "assignee": "Имя ответственного или null",
    "deadline": "Дедлайн или null",
    "priority": "high|medium|low"
  }
]"""


# ─── LLM ──────────────────────────────────────────────────────────────────────

def _get_llm_client() -> tuple["OpenAI", str]:
    """Создаёт OpenAI-compatible клиент из переменных окружения."""
    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
    api_key  = os.environ.get("LLM_API_KEY",  "not-required")
    model    = os.environ.get("LLM_MODEL",     "qwen3-32b")

    # В Docker-контейнере localhost — это сам контейнер, а не хост.
    # Авто-заменяем на host.docker.internal, если запущены внутри Docker.
    if os.path.exists("/.dockerenv") and re.search(r"localhost|127\.0\.0\.1", base_url):
        base_url = re.sub(r"localhost|127\.0\.0\.1", "host.docker.internal", base_url)
        print(f"[INFO] Docker detected: LLM_BASE_URL → {base_url}")

    return OpenAI(base_url=base_url, api_key=api_key), model


def _chat(client: "OpenAI", model: str, system: str, user: str) -> str:
    """Выполняет один запрос к LLM и возвращает текст ответа."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ─── Утилиты ──────────────────────────────────────────────────────────────────

def _chunk_transcript(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    """
    Разбивает транскрипт на части по границам абзацев (сегментов спикеров).
    Каждый чанк не превышает max_chars символов.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # +2 за \n\n
        if current_len + para_len > max_chars and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _apply_speaker_map(text: str, speaker_map: dict) -> str:
    """Заменяет SPEAKER_XX метки на реальные имена из speaker_map."""
    for code, name in speaker_map.items():
        text = text.replace(code, name)
    return text


def _extract_json_from_response(raw: str) -> str:
    """Извлекает JSON из ответа модели (убирает markdown code block если есть)."""
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
        if m:
            return m.group(1)
    return raw


# ─── Основная логика ──────────────────────────────────────────────────────────

def summarize(speakers_file: Path) -> None:
    output_dir = speakers_file.parent
    stem = speakers_file.name.replace("_speakers.txt", "")

    # ── Читаем транскрипт ─────────────────────────────────────────────────────
    transcript = speakers_file.read_text(encoding="utf-8")
    if not transcript.strip():
        print("[WARN] Файл транскрипта пуст, пропускаем.", file=sys.stderr)
        return

    # ── Применяем speaker_map.json (если есть) ────────────────────────────────
    speaker_map_path = output_dir / "speaker_map.json"
    if speaker_map_path.exists():
        try:
            speaker_map = json.loads(speaker_map_path.read_text(encoding="utf-8"))
            transcript = _apply_speaker_map(transcript, speaker_map)
            print(f"[INFO] speaker_map.json применён ({len(speaker_map)} спикеров)")
        except Exception as e:
            print(f"[WARN] Ошибка чтения speaker_map.json: {e}", file=sys.stderr)

    # ── LLM клиент ────────────────────────────────────────────────────────────
    client, model = _get_llm_client()
    print(f"[INFO] LLM: {model}")

    # ── MAP: summary по каждому чанку ─────────────────────────────────────────
    chunks = _chunk_transcript(transcript)
    print(f"[INFO] Чанков: {len(chunks)}")

    chunk_summaries: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        print(f"[INFO] Summary чанк {i}/{len(chunks)}...")
        s = _chat(
            client, model,
            SYSTEM_CHUNK_SUMMARY,
            f"Фрагмент {i} из {len(chunks)}:\n\n{chunk}",
        )
        chunk_summaries.append(s)

    # ── REDUCE: финальный summary ─────────────────────────────────────────────
    if len(chunk_summaries) == 1:
        reduce_input = chunk_summaries[0]
    else:
        reduce_input = "\n\n---\n\n".join(
            f"### Фрагмент {i}\n{s}" for i, s in enumerate(chunk_summaries, 1)
        )

    print("[INFO] Финальный summary (reduce)...")
    final_summary = _chat(client, model, SYSTEM_FINAL_SUMMARY,
                          f"Материалы встречи:\n\n{reduce_input}")

    summary_path = output_dir / f"{stem}_summary.md"
    summary_path.write_text(final_summary, encoding="utf-8")
    print(f"[OK]   {summary_path.name}")

    # ── ACTIONS: извлечение поручений ─────────────────────────────────────────
    # Источник — объединённые chunk-резюме: охватывают всю встречу, компактны
    print("[INFO] Извлечение action items...")
    actions_raw = _chat(client, model, SYSTEM_ACTIONS,
                        f"Материалы встречи:\n\n{reduce_input}")

    try:
        actions = json.loads(_extract_json_from_response(actions_raw))
    except json.JSONDecodeError:
        print("[WARN] JSON не распознан, сохраняем raw-ответ", file=sys.stderr)
        actions = [{"raw": actions_raw, "parse_error": True}]

    actions_path = output_dir / f"{stem}_actions.json"
    actions_path.write_text(
        json.dumps(actions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK]   {actions_path.name}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM summary pipeline для транскриптов встреч"
    )
    parser.add_argument(
        "--speakers-file",
        required=True,
        help="Путь к *_speakers.txt файлу (результат postprocess.py)",
    )
    args = parser.parse_args()

    path = Path(args.speakers_file)
    if not path.exists():
        print(f"[ERROR] Файл не найден: {path}", file=sys.stderr)
        sys.exit(1)

    summarize(path)


if __name__ == "__main__":
    main()
