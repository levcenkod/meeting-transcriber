#!/usr/bin/env python3
"""
evidence.py — проверка цитат и вычисление таймкодов кодом, не моделью.

Почему так (замеры на реальных данных, см. tz_v0_review.md):
  - точный substring-матч цитат теряет ~24 % верных пунктов: модель «чинит»
    кривой ASR («Мне зафиксировано» → «Я зафиксировал»), меняет пунктуацию;
  - t_sec от модели в 38 % случаев не попадает даже в блок цитаты (до ±75 с).

Поэтому: цитата ищется якорно (нормализация + окно ≥4 слов подряд),
а таймкод берётся из данных транскрипта по позиции найденной цитаты:
из пословных таймингов WhisperX (точно) или из заголовков блоков
*_speakers.txt (грубее, но всегда есть).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

MIN_ANCHOR_WORDS = 4          # минимальное окно совпадения
_WORD_RE = re.compile(r"[а-яёa-z0-9@€$%+]+", re.IGNORECASE)
_BLOCK_HEADER_RE = re.compile(
    r"\[(\d{1,2}):(\d{2}):(\d{2})\s*-\s*\d{1,2}:\d{2}:\d{2}\]\s*\S+:")


def _norm_word(w: str) -> str:
    return w.lower().replace("ё", "е")


def _tokenize(text: str) -> list[tuple[str, int]]:
    """[(нормализованное_слово, смещение_в_тексте), …]"""
    return [(_norm_word(m.group(0)), m.start()) for m in _WORD_RE.finditer(text)]


def _find_token_run(quote_tokens: list[str],
                    hay_tokens: list[str]) -> tuple[int, int] | None:
    """Найти самое длинное окно подряд идущих слов цитаты в тексте.

    Возвращает (позиция_в_hay, длина_окна) или None, если лучшее окно
    короче MIN_ANCHOR_WORDS (для коротких цитат — короче len(цитаты)).
    """
    if not quote_tokens or not hay_tokens:
        return None
    need = min(MIN_ANCHOR_WORDS, len(quote_tokens))
    best: tuple[int, int] | None = None

    # индекс первого слова каждого возможного окна цитаты
    hay_index: dict[str, list[int]] = {}
    for i, tok in enumerate(hay_tokens):
        hay_index.setdefault(tok, []).append(i)

    for q_start in range(len(quote_tokens)):
        first = quote_tokens[q_start]
        for h_start in hay_index.get(first, ()):
            length = 0
            while (q_start + length < len(quote_tokens)
                   and h_start + length < len(hay_tokens)
                   and quote_tokens[q_start + length] == hay_tokens[h_start + length]):
                length += 1
            if length >= need and (best is None or length > best[1]):
                best = (h_start, length)
        if best and best[1] == len(quote_tokens) - q_start:
            break  # длиннее уже не будет
    return best


def find_quote(quote: str, transcript: str) -> int | None:
    """Позиция (смещение в символах) якорного совпадения цитаты, или None."""
    q = _tokenize(quote or "")
    if not q:
        return None
    hay = _tokenize(transcript)
    hit = _find_token_run([t for t, _ in q], [t for t, _ in hay])
    if hit is None:
        return None
    return hay[hit[0]][1]


# ─── Таймкоды ────────────────────────────────────────────────────────────────

def _block_times(transcript: str) -> list[tuple[int, int]]:
    """[(смещение_начала_блока, секунда_начала), …] из заголовков *_speakers.txt."""
    out = []
    for m in _BLOCK_HEADER_RE.finditer(transcript):
        h, mnt, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        out.append((m.start(), h * 3600 + mnt * 60 + s))
    return out


def t_sec_from_blocks(char_pos: int, transcript: str) -> int | None:
    """Секунда начала блока, в котором лежит позиция цитаты."""
    best = None
    for off, sec in _block_times(transcript):
        if off <= char_pos:
            best = sec
        else:
            break
    return best


def _whisperx_words(wx: dict) -> list[tuple[str, float]]:
    """[(нормализованное_слово, start_sec), …] из WhisperX-json."""
    out = []
    for seg in wx.get("segments") or []:
        words = seg.get("words") or []
        if words:
            for w in words:
                token = _norm_word(str(w.get("word", "")).strip(" ,.!?—-…:;\"'«»"))
                start = w.get("start", seg.get("start"))
                if token and start is not None:
                    out.append((token, float(start)))
        else:
            for m in _WORD_RE.finditer(str(seg.get("text", ""))):
                if seg.get("start") is not None:
                    out.append((_norm_word(m.group(0)), float(seg["start"])))
    return out


def t_sec_from_whisperx(quote: str, wx_json_path: Path) -> int | None:
    """Секунда первого слова цитаты по пословным таймингам WhisperX."""
    try:
        wx = json.loads(wx_json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    words = _whisperx_words(wx)
    q = [t for t, _ in _tokenize(quote or "")]
    hit = _find_token_run(q, [t for t, _ in words])
    if hit is None:
        return None
    return int(words[hit[0]][1])


def verify_item(quote: str, transcript: str,
                wx_json_path: Path | None = None) -> tuple[bool, int | None]:
    """(цитата_подтверждена, t_sec | None) — одна точка входа для пайплайна."""
    pos = find_quote(quote, transcript)
    if pos is None:
        return False, None
    t = None
    if wx_json_path is not None and wx_json_path.exists():
        t = t_sec_from_whisperx(quote, wx_json_path)
    if t is None:
        t = t_sec_from_blocks(pos, transcript)
    return True, t


# ─── Токены анонимизации, которые модель выдумала ────────────────────────────

_ANON_TOKEN_RE = re.compile(
    r"\b(?:PERSON|COMPANY|LOCATION|EMAIL|PHONE|URL|DOMAIN|TRANSACTION)_\d{3}\b")


def unresolved_tokens(text: str, known_tokens: set[str]) -> set[str]:
    """Токены вида PERSON_014 в тексте, которых нет в карте анонимизации."""
    if not text:
        return set()
    return {t for t in _ANON_TOKEN_RE.findall(text) if t not in known_tokens}
