#!/usr/bin/env python3
"""
anonymize.py -- Local PII anonymization for meeting transcripts.

Replaces sensitive entities with deterministic tokens before sending to online LLM.
The real-to-token mapping is stored ONLY locally and never sent outside.

Usage as module:
    from anonymize import Anonymizer
    anon = Anonymizer(map_path, language="ru")
    anon_text = anon.anonymize(chunk_text)
    real_result = anon.deanonymize_any(llm_json_response)

Entity types detected:
    PERSON       -> PERSON_001      (spaCy NER: PER)
    COMPANY      -> COMPANY_001     (spaCy NER: ORG)
    LOCATION     -> LOCATION_001    (spaCy NER: LOC/GPE)
    EMAIL        -> EMAIL_001       (regex)
    PHONE        -> PHONE_001       (regex)
    URL          -> URL_001         (regex)
    DOMAIN       -> DOMAIN_001      (regex)
    TRANSACTION  -> TRANSACTION_001 (regex)
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Regex patterns (applied before spaCy; ordered most-specific first)
# ---------------------------------------------------------------------------

_REGEX_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(
        r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'
    )),
    ("URL", re.compile(
        r'https?://[^\s\])"\'>\u00bb]+'
    )),
    ("DOMAIN", re.compile(
        r'(?<![/@\w])'
        r'(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)'
        r'+(?:com|org|net|io|ru|de|uk|fr|es|it|co|gov|edu|biz|info|ai)\b',
        re.IGNORECASE,
    )),
    ("PHONE", re.compile(
        r'(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}'
        r'|\+\d{1,3}[\s\-]\d{2,4}[\s\-]\d{3,4}[\s\-]\d{3,4}'
    )),
    ("TRANSACTION", re.compile(
        r'\b[A-Z]{2,6}[-_]?\d{6,}\b'
        r'|\b\d{12,}\b'
    )),
]

# spaCy label -> our token prefix
_SPACY_LABEL_MAP: dict[str, str | None] = {
    "PER":    "PERSON",
    "PERSON": "PERSON",
    "ORG":    "COMPANY",
    "LOC":    "LOCATION",
    "GPE":    "LOCATION",
    "FAC":    "LOCATION",
    "MISC":   None,
}

# Words that should NEVER be anonymized even if spaCy tags them as entities
_SKIP_WORDS: frozenset[str] = frozenset({
    # Tech
    "docker", "kubernetes", "redis", "postgresql", "postgres", "nginx",
    "linux", "ubuntu", "python", "javascript", "typescript", "react",
    "vue", "http", "https", "api", "rest", "json", "xml", "sql",
    "git", "github", "gitlab", "slack", "jira", "confluence",
    # Time
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december",
    "понедельник", "вторник", "среда", "четверг", "пятница",
    "суббота", "воскресенье",
    "январь", "февраль", "март", "апрель", "май", "июнь",
    "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь",
    # Common
    "ok", "yes", "no", "да", "нет", "ок", "info", "error", "warn",
})

# Preferred spaCy models per language (tried in order)
_SPACY_MODELS: dict[str, list[str]] = {
    "ru": ["ru_core_news_md", "ru_core_news_sm", "xx_ent_wiki_sm"],
    "en": ["en_core_web_sm", "en_core_web_md", "xx_ent_wiki_sm"],
}
_SPACY_FALLBACK = ["xx_ent_wiki_sm", "ru_core_news_sm", "en_core_web_sm"]


def _load_spacy(language: str = "ru"):
    """Try to load a spaCy NER model. Returns None if unavailable."""
    try:
        import spacy
    except ImportError:
        return None
    models = list(dict.fromkeys(
        _SPACY_MODELS.get(language, []) + _SPACY_FALLBACK
    ))
    for name in models:
        try:
            nlp = spacy.load(name, disable=["parser", "tagger", "morphologizer", "senter"])
            return nlp
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Anonymizer
# ---------------------------------------------------------------------------

class Anonymizer:
    """
    Deterministic local anonymizer.

    All entity->token mappings are stored in map_path (JSON) and
    are NEVER sent to external services.
    """

    def __init__(self, map_path: Path, language: str = "ru", enabled: bool = True):
        self._map: dict[str, str] = {}        # token -> real value
        self._reverse: dict[str, str] = {}    # real value -> token
        self._counters: dict[str, int] = {}   # prefix -> last used number
        self._map_path = map_path
        self._enabled = enabled
        self._nlp = None

        if enabled:
            self._nlp = _load_spacy(language)
            if self._nlp:
                print(f"[INFO] Anonymizer NER: {self._nlp.meta.get('name', 'unknown')}")
            else:
                print("[WARN] Anonymizer: spaCy model not found -- using regex only",
                      file=sys.stderr)

        # Restore existing map (for cross-chunk consistency)
        if map_path.exists():
            try:
                existing = json.loads(map_path.read_text(encoding="utf-8"))
                self._map = existing
                self._reverse = {v: k for k, v in existing.items()}
                for token in existing:
                    parts = token.rsplit("_", 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        prefix = parts[0]
                        num = int(parts[1])
                        self._counters[prefix] = max(
                            self._counters.get(prefix, 0), num
                        )
            except Exception as e:
                print(f"[WARN] Could not restore anonymization map: {e}",
                      file=sys.stderr)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_token(self, entity_type: str, value: str) -> str:
        """Return existing token or create a new deterministic one."""
        if value in self._reverse:
            return self._reverse[value]
        self._counters[entity_type] = self._counters.get(entity_type, 0) + 1
        token = f"{entity_type}_{self._counters[entity_type]:03d}"
        self._map[token] = value
        self._reverse[value] = token
        return token

    def _save_map(self) -> None:
        self._map_path.write_text(
            json.dumps(self._map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _env_allows(self, entity_type: str) -> bool:
        """Check if this entity type is enabled via env var."""
        key = f"ANONYMIZE_{entity_type}S"
        return os.environ.get(key, "true").lower() in ("true", "1", "yes")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def anonymize(self, text: str) -> str:
        """
        Replace sensitive entities in text with tokens.
        Updates and saves the map file.
        Returns anonymized text.
        """
        if not self._enabled:
            return text

        # Collect (start, end, token) tuples; avoid overlapping spans
        replacements: list[tuple[int, int, str]] = []
        covered: set[tuple[int, int]] = set()

        def _add(start: int, end: int, etype: str, value: str) -> None:
            if any(s < end and start < e for s, e in covered):
                return
            token = self._get_token(etype, value)
            replacements.append((start, end, token))
            covered.add((start, end))

        # Step 1: regex patterns
        for entity_type, pattern in _REGEX_PATTERNS:
            if not self._env_allows(entity_type):
                continue
            for m in pattern.finditer(text):
                _add(m.start(), m.end(), entity_type, m.group())

        # Step 2: spaCy NER
        if self._nlp:
            doc = self._nlp(text)
            for ent in doc.ents:
                etype = _SPACY_LABEL_MAP.get(ent.label_)
                if not etype:
                    continue
                if not self._env_allows(etype):
                    continue
                if len(ent.text.strip()) < 3:
                    continue
                if ent.text.lower() in _SKIP_WORDS:
                    continue
                _add(ent.start_char, ent.end_char, etype, ent.text)

        if not replacements:
            return text

        # Apply in reverse order to keep offsets valid
        replacements.sort(key=lambda x: x[0], reverse=True)
        chars = list(text)
        for start, end, token in replacements:
            chars[start:end] = list(token)

        self._save_map()
        return "".join(chars)

    def deanonymize_text(self, text: str) -> str:
        """Replace all tokens in text with original values."""
        if not self._map:
            return text
        # Longest tokens first to avoid partial replacements
        for token, real in sorted(self._map.items(),
                                  key=lambda x: len(x[0]), reverse=True):
            text = text.replace(token, real)
        return text

    def deanonymize_any(self, obj: Any) -> Any:
        """Recursively deanonymize any JSON-compatible structure."""
        if isinstance(obj, str):
            return self.deanonymize_text(obj)
        if isinstance(obj, list):
            return [self.deanonymize_any(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self.deanonymize_any(v) for k, v in obj.items()}
        return obj

    @property
    def map(self) -> dict[str, str]:
        """Read-only view of the token->real mapping."""
        return dict(self._map)


# ---------------------------------------------------------------------------
# CLI (standalone usage)
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Anonymize a *_speakers.txt file"
    )
    parser.add_argument("--input",    required=True, help="Path to *_speakers.txt")
    parser.add_argument("--map-out",  help="Output path for anonymization map JSON")
    parser.add_argument("--language", default=os.environ.get("LANGUAGE", "ru"))
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)

    stem = path.stem.replace("_speakers", "")
    map_path = (
        Path(args.map_out) if args.map_out
        else path.parent / f"{stem}_anonymization_map.json"
    )

    text = path.read_text(encoding="utf-8")
    anon = Anonymizer(map_path=map_path, language=args.language)
    result = anon.anonymize(text)

    out_path = path.parent / path.name.replace("_speakers.txt", "_anonymized.txt")
    out_path.write_text(result, encoding="utf-8")
    print(f"[OK] {out_path.name}")
    print(f"[OK] Map: {map_path.name}  ({len(anon.map)} entities)")


if __name__ == "__main__":
    main()