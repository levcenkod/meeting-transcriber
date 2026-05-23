#!/usr/bin/env python3
"""
postprocess.py — создаёт transcript_speakers.txt из JSON-результата WhisperX.

Формат вывода:
    [00:00:01 - 00:00:08] SPEAKER_00:
    Всем привет, давайте начнём.

    [00:00:09 - 00:00:20] SPEAKER_01:
    Я обновлю статус по логистике.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def format_time(seconds) -> str:
    """Конвертирует секунды в строку вида HH:MM:SS. None/невалидное → 00:00:00."""
    try:
        seconds = max(0.0, float(seconds))
    except (TypeError, ValueError):
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_speaker_transcript(data: dict) -> str:
    """Строит текст с группировкой по спикерам из JSON WhisperX."""
    segments = data.get("segments", [])
    if not segments:
        return "(Нет сегментов в JSON)\n"

    lines = []
    current_speaker = None
    current_start = None
    current_end = None
    current_words: list[str] = []

    def flush_segment():
        if current_words:
            time_tag = f"[{format_time(current_start)} - {format_time(current_end)}]"
            speaker_label = current_speaker if current_speaker else "UNKNOWN_SPEAKER"
            lines.append(f"{time_tag} {speaker_label}:")
            lines.append(" ".join(current_words))
            lines.append("")

    def _num(v, fallback=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return fallback

    for seg in segments:
        start = _num(seg.get("start"))
        end = _num(seg.get("end"), start)
        text = (seg.get("text") or "").strip()
        speaker = seg.get("speaker")  # может отсутствовать

        if speaker != current_speaker:
            flush_segment()
            current_speaker = speaker
            current_start = start
            current_end = end
            current_words = [text] if text else []
        else:
            current_end = end
            if text:
                current_words.append(text)

    flush_segment()

    return "\n".join(lines)


def find_json_file(output_dir: Path, stem: str) -> Path | None:
    """Ищет JSON файл WhisperX по имени (без расширения)."""
    candidates = [
        output_dir / f"{stem}.json",
        output_dir / f"{stem}.words.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Ищем любой JSON с таким stem (на случай нестандартного именования)
    for p in output_dir.glob(f"{stem}*.json"):
        return p

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Постобработка JSON-результата WhisperX → transcript_speakers.txt"
    )
    parser.add_argument(
        "--output-dir", default="/output", help="Папка с результатами WhisperX"
    )
    parser.add_argument(
        "--filename", required=True, help="Имя файла без расширения (stem)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    stem = args.filename

    if not output_dir.exists():
        print(f"[ERROR] Папка не найдена: {output_dir}", file=sys.stderr)
        sys.exit(1)

    json_path = find_json_file(output_dir, stem)
    if json_path is None:
        print(
            f"[ERROR] JSON-файл для '{stem}' не найден в {output_dir}. "
            "Убедитесь, что WhisperX завершился успешно.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[INFO] Читаем JSON: {json_path}")

    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Не удалось разобрать JSON: {e}", file=sys.stderr)
        sys.exit(1)

    transcript = build_speaker_transcript(data)

    out_path = output_dir / f"{stem}_speakers.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    print(f"[OK] Файл создан: {out_path}")


if __name__ == "__main__":
    main()
