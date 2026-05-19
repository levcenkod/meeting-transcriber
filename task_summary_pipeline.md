# Задача: Summary Pipeline

## Статус: ✅ Реализовано

---

## Требования

1. [x] Создать `scripts/summarize.py`
2. [x] Скрипт принимает путь к `*_speakers.txt`
3. [x] Читает `speaker_map.json`, если он есть рядом с файлом
4. [x] Заменяет `SPEAKER_00` / `SPEAKER_01` на реальные имена
5. [x] Создаёт `<stem>_summary.md`
6. [x] Создаёт `<stem>_actions.json`
7. [x] LLM endpoint вынести в `.env`:
   - `LLM_PROVIDER=openai_compatible`
   - `LLM_BASE_URL=http://localhost:1234/v1`
   - `LLM_API_KEY=not-required`
   - `LLM_MODEL=qwen3-32b`
8. [x] Использовать OpenAI-compatible API формат (`openai` пакет)
9. [x] Если transcript большой — разбивать на chunks (`CHUNK_MAX_CHARS = 12 000`)
10. [x] Map-reduce:
    - summary по каждому чанку (map)
    - финальный общий summary (reduce)
11. [x] После транскрибации автоматически запускать `summarize.py`

---

## Архитектура

```
*_speakers.txt
    │
    ├── [speaker_map.json?] → replace SPEAKER_XX → real names
    │
    ├── chunk_transcript()          # разбивка по ~3000 токенов
    │
    ├── MAP: LLM chunk summary x N
    │
    ├── REDUCE: LLM final summary   → <stem>_summary.md
    │
    └── ACTIONS: LLM extract        → <stem>_actions.json
```

### Map prompt
Краткое резюме фрагмента. Язык — оригинал транскрипта.

### Reduce prompt
Финальный Markdown с разделами:
- `## Краткое резюме`
- `## Основные темы`
- `## Ключевые решения`

### Actions prompt
JSON-массив: `[{ action, assignee, deadline, priority }]`  
Источник: объединённые chunk-резюме (покрывают всю встречу).

---

## Конфигурация (.env)

| Переменная      | По умолчанию                    | Описание                     |
|-----------------|----------------------------------|------------------------------|
| `LLM_PROVIDER`  | `openai_compatible`             | Тип провайдера               |
| `LLM_BASE_URL`  | `http://localhost:1234/v1`      | URL API (LM Studio, Ollama…) |
| `LLM_API_KEY`   | `not-required`                  | API ключ (если нужен)        |
| `LLM_MODEL`     | `qwen3-32b`                     | Имя модели                   |

### Запуск в Docker
Скрипт авто-определяет Docker (`/.dockerenv`) и заменяет `localhost` →  
`host.docker.internal`, чтобы достучаться до LM Studio на хосте.

---

## Файлы

| Файл                        | Изменение                               |
|-----------------------------|------------------------------------------|
| `scripts/summarize.py`      | Создан (новый)                          |
| `.env.example`              | Добавлен блок LLM-переменных            |
| `Dockerfile`                | `pip install openai` + COPY summarize.py|
| `transcribe.ps1`            | Шаг summarize после postprocess         |

---

## speaker_map.json (формат)

```json
{
  "SPEAKER_00": "Иван Петров",
  "SPEAKER_01": "Мария Сидорова"
}
```

Файл кладётся рядом с `*_speakers.txt` вручную или автоматически.

---

## Следующие задачи (backlog)

- [ ] Авто-идентификация спикеров (voice embedding → имя)
- [ ] Web UI для просмотра summary + actions
- [ ] Отправка actions в Jira / Notion / Telegram
