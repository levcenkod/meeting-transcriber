# Финальное ТЗ: Meeting Intelligence Pipeline для планёрок (Simplified Production Version)

## Цель

Построить локальный pipeline для обработки записей планёрок:

```text
Аудио/видео запись встречи
↓
WhisperX + diarization
↓
transcript со спикерами и таймкодами
↓
smart chunking
↓
structured extraction
↓
merge + deduplication
↓
evidence check
↓
final summary
↓
Markdown summary + JSON artifacts
```

Главная цель — не просто сделать краткое summary, а получить проверяемые и полезные результаты:

- что обсуждали;
- какие решения приняли;
- какие задачи появились;
- кто ответственный;
- какие дедлайны;
- какие риски и блокеры;
- какие вопросы остались открытыми;
- где в transcript есть подтверждение каждому важному выводу.

---

# Почему простой map-reduce недостаточен

Простой flow:

```text
transcript
↓
chunks по 12 000 символов
↓
summary каждого чанка
↓
summary of summaries
↓
actions из summary
```

может терять важный контекст.

Основные проблемы:

1. Потеря связей между частями встречи.
2. Потеря коротких поручений.
3. Резка transcript посередине реплик.
4. Потеря деталей при summary of summaries.

---

# Основной принцип

Нужно делать не `summary-first`, а `evidence-first`.

Плохо:

```text
raw transcript
↓
summary
↓
actions
```

Хорошо:

```text
raw transcript
↓
facts / actions / decisions / risks with evidence
↓
merge + validation
↓
final summary
```

Transcript должен оставаться источником истины.

---

# Целевая архитектура

```text
1. Transcription layer
   WhisperX
   → *_speakers.txt
   → *_segments.json

2. Processing layer
   speaker_map.json
   → readable transcript with real names

3. Chunking layer
   smart chunks by speaker blocks
   → overlap 10–20%

4. Map extraction layer
   raw chunk
   → structured JSON

5. Merge layer
   all chunk JSONs
   → merged analysis

6. Deduplication layer
   repeated actions/decisions/risks
   → single normalized entities

7. Evidence layer
   each important item
   → source_time + evidence quote

8. Final summary layer
   structured data
   → final summary.md

9. Artifacts layer
   *_summary.md
   *_actions.json
   *_decisions.json
   *_analysis.json
```

---

# Артефакты на выходе

Для одной встречи должны создаваться:

```text
meeting.mp3
meeting_speakers.txt
meeting_segments.json
meeting_analysis.json
meeting_summary.md
meeting_actions.json
meeting_decisions.json
```

---

# speaker_map.json

Если рядом с transcript есть `speaker_map.json`, его нужно применить.

```json
{
  "SPEAKER_00": "Алексей",
  "SPEAKER_01": "Георгий",
  "SPEAKER_02": "Дмитрий"
}
```

Если speaker не найден в map, оставить исходное значение.

---

# Smart chunking

## Главное правило

Нельзя резать transcript посередине speaker-блока.

Хорошо:

```text
chunk = набор целых speaker-блоков
```

---

## Базовые настройки

```env
CHUNK_SIZE_CHARS=12000
CHUNK_OVERLAP_CHARS=2000
SPLIT_BY_SPEAKER_BLOCKS=true
```

---

## Overlap

Overlap нужен, чтобы не терять контекст на границах чанков.

Пример:

```text
Chunk 1:
blocks 1–40

Chunk 2:
blocks 35–75

Chunk 3:
blocks 70–110
```

Overlap должен быть по speaker-блокам, а не тупо по символам.

---

# Structured Map Extraction

Для каждого чанка LLM должна возвращать строго валидный JSON.

Не нужно просить модель просто “сделай summary”.

Нужно извлечь:

- summary;
- decisions;
- action_items;
- blockers;
- risks;
- open_questions;
- important_facts;
- mentioned_topics.

---

# Map JSON schema

```json
{
  "chunk_id": 1,
  "time_range": {
    "start": "00:00:01",
    "end": "00:12:45"
  },
  "summary": "Краткое резюме этого фрагмента.",
  "decisions": [
    {
      "decision": "Что решили",
      "context": "Почему это решили",
      "speakers": ["Алексей", "Георгий"],
      "source_time": "00:04:12",
      "evidence": "Короткая цитата/фрагмент, подтверждающий решение",
      "confidence": "high"
    }
  ],
  "action_items": [
    {
      "task": "Что нужно сделать",
      "owner": "Георгий",
      "deadline": "2026-05-20",
      "context": "Контекст поручения",
      "source_time": "00:10:14",
      "evidence": "Георгий, глянь завтра логи ALB по этому callback",
      "confidence": "medium"
    }
  ],
  "blockers": [],
  "risks": [],
  "open_questions": [],
  "important_facts": [],
  "mentioned_topics": []
}
```

---

# Правила для extraction

- Не выдумывать факты.
- Не выдумывать ответственных.
- Не выдумывать дедлайны.
- Если owner неизвестен — `owner = null`.
- Если deadline неизвестен — `deadline = null`.
- Если уверенность низкая — `confidence = "low"`.
- Для каждого важного элемента указывать `source_time`, если есть таймкод.
- Для каждого важного элемента указывать `evidence`, если возможно.
- Если это просто обсуждение — не записывать как decision.
- Если это не поручение — не записывать как action item.

---

# Actions extraction

Action items нужно извлекать напрямую из raw chunks, а не из summary.

Плохо:

```text
chunk → summary → actions
```

Хорошо:

```text
chunk → action_items[]
```

Причина: короткие поручения часто теряются при summary.

---

# Merge step

После обработки всех чанков нужно собрать единый объект:

```json
{
  "meeting_id": "meeting_2026_05_19",
  "chunk_summaries": [],
  "decisions": [],
  "action_items": [],
  "blockers": [],
  "risks": [],
  "open_questions": [],
  "important_facts": [],
  "mentioned_topics": []
}
```

---

# Deduplication

Из-за overlap некоторые элементы будут повторяться.

Нужно дедуплицировать:

- action_items;
- decisions;
- blockers;
- risks;
- open_questions.

---

# Evidence check

Каждый важный вывод должен иметь подтверждение:

```json
{
  "source_time": "00:10:14",
  "evidence": "короткий фрагмент из transcript"
}
```

Если evidence нет:
- либо попытаться найти его в transcript;
- либо поставить `confidence = "low"`.

---

# Final summary generation

Финальный summary должен строиться не из summary чанков, а из merged structured data.

На вход summary generation передаётся:

```text
- chunk_summaries
- merged decisions
- merged action_items
- merged blockers
- merged risks
- merged open_questions
- important_facts
- mentioned_topics
```

На выходе:

```text
*_summary.md
```

---

# Итоговый summary.md

```md
# Резюме планёрки

## Кратко

2–5 предложений: о чём была встреча и главный итог.

## Основные темы

- Тема 1
- Тема 2
- Тема 3

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
```

---

# Итоговый actions.json

```json
[
  {
    "task": "Проверить payout callback",
    "owner": "Георгий",
    "deadline": "2026-05-20",
    "context": "Нужно проверить ALB logs и Horizon job по проблемной выплате",
    "source_times": ["00:10:14", "00:41:22"],
    "evidence": [
      "Георгий, глянь завтра логи ALB по этому callback"
    ],
    "confidence": "high",
    "status": "open"
  }
]
```

---

# Итоговый decisions.json

```json
[
  {
    "decision": "Оставить текущую схему обработки callback, но добавить проверку ALB logs",
    "context": "Обсуждали расхождение между Horizon job и merchant callback",
    "speakers": ["Алексей", "Георгий"],
    "source_times": ["00:14:02"],
    "evidence": [
      "Ок, тогда пока схему не меняем, но ALB надо проверить"
    ],
    "confidence": "high"
  }
]
```

---

# Новый flow в summarize.py

```text
main()
↓
read *_speakers.txt
↓
load speaker_map.json if exists
↓
apply speaker map
↓
parse speaker blocks
↓
build chunks with overlap
↓
for each chunk:
    call LLM with structured extraction prompt
    validate JSON
    save intermediate result to /intermediate/chunk_N.json
↓
merge chunk results
↓
deduplicate merged results
↓
evidence check
↓
save *_analysis.json
↓
generate final summary
↓
save *_summary.md
↓
save *_actions.json
↓
save *_decisions.json
```

---

# Environment settings

```env
CHUNK_SIZE_CHARS=12000
CHUNK_OVERLAP_CHARS=2000
SPLIT_BY_SPEAKER_BLOCKS=true

MAP_OUTPUT=json
ACTIONS_FROM_RAW_CHUNKS=true
FINAL_SUMMARY_FROM_STRUCTURED_DATA=true

ENABLE_EVIDENCE_CHECK=true

LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=local-or-real-key
LLM_MODEL=qwen3-32b
```

---

# Map prompt

```text
Ты анализируешь фрагмент транскрипта рабочей планёрки.

Твоя задача — не просто сделать краткое резюме, а извлечь структурированные факты.

Верни строго валидный JSON без Markdown.

Извлеки:
- краткое summary фрагмента
- принятые решения
- поручения/action items
- блокеры
- риски
- открытые вопросы
- важные факты
- темы, которые обсуждались

Правила:
- Не выдумывай то, чего нет в тексте.
- Если ответственный не указан — owner = null.
- Если дедлайн не указан — deadline = null.
- Если уверенность низкая — confidence = "low".
- Для каждого важного элемента укажи source_time, если таймкод есть.
- Для каждого важного элемента укажи evidence — короткий фрагмент transcript, подтверждающий вывод.
- Сохраняй имена спикеров, если они есть.
- Если это просто обсуждение без решения — не записывай его как decision.
- Если это не поручение — не записывай как action item.
```

---

# Reduce prompt

```text
Ты получил структурированный анализ всех фрагментов рабочей планёрки.

На основе этих данных сформируй финальное Markdown-резюме встречи.

Правила:
- Не выдумывай новые решения, задачи или дедлайны.
- Используй только предоставленные structured data.
- Если ответственный неизвестен — пиши "не указан".
- Если дедлайн неизвестен — пиши "не указан".
- Сгруппируй похожие темы.
- Укажи action items в таблице.
- Укажи решения, риски, блокеры и открытые вопросы.
- Для важных пунктов добавь короткое подтверждение/evidence или source_time.
```

---

# Критерии успеха

Pipeline считается качественным, если:

1. Transcript разбивается на чанки без разрыва speaker-блоков.
2. Есть overlap между чанками.
3. Каждый Map-запрос возвращает structured JSON.
4. Actions извлекаются из raw chunks, а не из summary.
5. Есть merge + deduplication.
6. Для важных элементов есть `source_time` и `evidence`.
7. Финальный summary строится из structured data.
8. `actions.json` содержит owner, deadline, context, confidence, evidence.
9. `decisions.json` отделён от `actions.json`.

---

# Рекомендуемый порядок реализации

## Этап 1

Сделать structured map extraction:

```text
chunk → JSON
```

## Этап 2

Сделать smart chunking по speaker-блокам + overlap.

## Этап 3

Сделать merge + deduplication.

## Этап 4

Сделать генерацию:

```text
*_analysis.json
*_actions.json
*_decisions.json
```

## Этап 5

Сделать финальный:

```text
*_summary.md
```

## Этап 6

Добавить evidence check.

---

# Итоговая оценка подхода

```text
Простой map-reduce: 6/10
Structured map-reduce: 8/10
Structured extraction + evidence check: 8.5/10
```

Главная идея:

```text
Не summary-first.
А evidence-first.
```

Для планёрок это важнее всего, потому что задачи, решения, ответственные и дедлайны могут быть разбросаны по разным частям встречи.
