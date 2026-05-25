# meeting-transcriber

Локальный Docker-проект для автоматической транскрибации аудио/видео записей встреч с разделением по спикерам (**WhisperX** + **pyannote**) и интеллектуальным анализом через LLM.

---

## Что делает проект

Вы загружаете аудио/видео файл — в папке `output/` появляются:

| Файл | Описание |
|---|---|
| `*_speakers.txt` | Транскрипт, сгруппированный по спикерам |
| `*_summary.md` | Резюме совещания в Markdown |
| `*_actions.json` | Поручения (кто, что, срок) |
| `*_decisions.json` | Принятые решения |
| `*_analysis.json` | Полная структурированная аналитика |
| `*.txt / *.srt / *.json` | Сырой транскрипт во всех форматах WhisperX |

Пример `*_speakers.txt`:

```
[00:00:01 - 00:00:08] SPEAKER_00:
Всем привет, давайте начнём.

[00:00:09 - 00:00:20] SPEAKER_01:
Я обновлю статус по логистике.
```

---

## Требования

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows, macOS, Linux)
- Токен Hugging Face (бесплатно)
- PowerShell 5.1+ — только для CLI-режима (встроен в Windows)
- Опционально: видеокарта NVIDIA с CUDA для ускорения транскрипции
- Опционально: LLM-сервер (LM Studio, Ollama, OpenAI) для анализа

---

## Установка

### 1. Установите Docker Desktop

Скачайте и установите [Docker Desktop](https://www.docker.com/products/docker-desktop/).

### 2. Включите WSL2 backend (Windows)

При установке Docker Desktop включите опцию **"Use WSL 2 based engine"** (рекомендуется).  
Проверить: `Settings → General → Use the WSL 2 based engine`.

### 3. Проверьте GPU (опционально)

Если у вас видеокарта NVIDIA:

```powershell
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Если команда выводит информацию о GPU — всё готово. Если ошибка — проект будет работать на CPU (медленнее).

Для GPU требуется [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### 4. Получите токен Hugging Face

1. Зарегистрируйтесь на [huggingface.co](https://huggingface.co)
2. Перейдите в [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Создайте токен с правами `read`

### 5. Примите условия использования моделей pyannote

Войдите в Hugging Face и примите условия на страницах (кнопка **"Agree and access repository"**):

- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

> ⚠️ Нужно принять оба — иначе diarization упадёт с ошибкой 403.

### 6. Настройте `.env`

```powershell
cd meeting-transcriber
copy .env.example .env
```

Откройте `.env` и укажите свой токен:

```env
HF_TOKEN=hf_ваш_реальный_токен
```

Остальные настройки можно оставить по умолчанию. Подробнее — в разделе [Конфигурация](#конфигурация).

---

## Запуск

### Способ 1 — Веб-интерфейс (рекомендуется)

Дважды кликните `start.bat`. Откроется терминал, контейнер запустится автоматически.  
Откройте браузер: **http://localhost:8080**

Что вы увидите:
- Перетащите аудио/видео файл в зону загрузки (или нажмите «Выбрать файл»)
- Выберите язык и категорию встречи
- Включите или выключите LLM-анализ
- Нажмите **«Начать транскрипцию»** — лог будет отображаться в реальном времени
- После завершения: скачайте результаты и прочитайте резюме прямо в браузере

Для остановки — закройте окно терминала или нажмите `Ctrl+C`.

### Способ 2 — Командная строка (PowerShell)

```powershell
# Один файл
.\transcribe.ps1 .\input\meeting.mp3

# Batch-режим — все файлы из папки
.\transcribe.ps1 .\input
```

Поддерживаемые форматы: `.mp3`, `.wav`, `.m4a`, `.ogg`, `.mp4`, `.mkv`, `.flac`, `.aac`, `.webm`

---

## Первый запуск

При первом запуске Docker скачает и соберёт образ (~5–15 минут в зависимости от скорости интернета и наличия GPU). Последующие запуски быстрее — образ и модели кэшируются.

---

## LLM-анализ

После транскрипции можно автоматически получить структурированный анализ через любой OpenAI-совместимый LLM.

Укажите в `.env`:

```env
LLM_BASE_URL=http://localhost:1234/v1   # LM Studio
LLM_API_KEY=not-required
LLM_MODEL=qwen3-32b
```

Что извлекает анализатор из каждого совещания:
- **Принятые решения** с контекстом и цитатами
- **Поручения** — кто, что и к какому сроку
- **Блокеры** и **риски**
- **Открытые вопросы**
- Общее **резюме**

Совместимые серверы: [LM Studio](https://lmstudio.ai), [Ollama](https://ollama.com), [OpenAI API](https://platform.openai.com).

---

## Анонимизация (опционально)

Если вы используете **онлайн LLM** (OpenAI и т.п.), включите локальную анонимизацию — имена людей, компании, email, телефоны и другие PII будут заменены токенами (`PERSON_001`, `COMPANY_001` …) перед отправкой в LLM и восстановлены в финальных файлах. Маппинг хранится только локально в `*_anonymization_map.json`.

Включить в `.env`:

```env
ENABLE_ANONYMIZATION=true
```

Тонкая настройка: `ANONYMIZE_PERSONS`, `ANONYMIZE_EMAILS`, `ANONYMIZE_PHONES` и другие флаги (см. `.env.example`).

---

## Интеграция с Obsidian

Результаты автоматически складываются в ваш Obsidian Vault как готовые Markdown-заметки — никаких плагинов разрабатывать не нужно.

### Настройка

1. В `.env` укажите абсолютный путь к корню вашего Vault и подпапку для планёрок:

   ```env
   # Windows
   OBSIDIAN_VAULT_PATH=C:/Users/me/Documents/MyVault
   # Linux/macOS
   # OBSIDIAN_VAULT_PATH=/home/me/Vaults/Work

   OBSIDIAN_SUBFOLDER=Meetings
   ```

   По умолчанию (если переменная не задана) используется локальная папка `./obsidian_vault` в корне проекта.

2. Перезапустите контейнер: `docker compose up -d`.

3. После первой успешной обработки в Vault появится:
   - `{OBSIDIAN_SUBFOLDER}/Dashboard.md` — главная страница со списком встреч и открытыми задачами (Dataview).
   - `{OBSIDIAN_SUBFOLDER}/<Категория>/<дата> — <название>.md` — заметка по каждой встрече.

В Web UI можно менять модель LLM и подпапку Obsidian «на лету» — настройки сохраняются в `output/_settings.json` и применяются к следующим запускам без перезапуска контейнера.

### Что внутри заметки

- **YAML frontmatter** с `title`, `date`, `category`, `attendees`, `tags`.
- **Mermaid mindmap** «Карта обсуждения» — Obsidian рендерит его нативно.
- **Action items** в формате Obsidian Tasks: `- [ ] Задача 👤 @Имя 📅 2025-01-15`.
- **Wiki-links** на ключевые сущности и участников: `[[Docker]]`, `[[Whisper]]`, `[[Иван Иванов]]`.

### Обязательные плагины Obsidian

Откройте ваш Vault и установите два community-плагина через **Settings → Community plugins → Browse**:

| Плагин | Зачем нужен |
|---|---|
| **Dataview** | Рендерит таблицу встреч и сводку категорий на `Dashboard.md`. |
| **Tasks** | Агрегирует все `- [ ]` пункты из заметок, фильтрует по дедлайну/исполнителю. |

После установки включите оба плагина и откройте `Dashboard.md` — таблицы и список задач заполнятся автоматически.

---

## Конфигурация

Все настройки — в файле `.env`. Полный пример с комментариями: `.env.example`.

| Переменная | По умолчанию | Описание |
|---|---|---|
| `HF_TOKEN` | — | Токен Hugging Face (обязательно) |
| `MODEL` | `large-v3` | Модель WhisperX |
| `LANGUAGE` | `ru` | Язык аудио |
| `DEVICE` | `cuda` | `cuda` или `cpu` |
| `COMPUTE_TYPE` | `float16` | `float16` (GPU) или `int8` (CPU) |
| `MIN_SPEAKERS` / `MAX_SPEAKERS` | — | Подсказка диаризатору |
| `LLM_BASE_URL` | — | URL LLM API |
| `LLM_MODEL` | `qwen3-32b` | Имя модели |
| `CHUNK_SIZE_CHARS` | `12000` | Размер чанка транскрипта |
| `ENABLE_ANONYMIZATION` | `false` | Анонимизация перед отправкой в LLM |

#### CPU fallback (нет GPU)

```env
DEVICE=cpu
COMPUTE_TYPE=int8
MODEL=medium
```

---

## Структура проекта

```
meeting-transcriber/
├── Dockerfile              # Docker-образ (WhisperX, PyTorch, Flask, spaCy)
├── docker-compose.yml      # Веб-интерфейс на порту 8080
├── start.bat               # Двойной клик — запуск веб-UI
├── transcribe.ps1          # CLI-скрипт (PowerShell)
├── app.py                  # Flask веб-сервер
├── templates/
│   └── index.html          # Веб-интерфейс
├── .env.example            # Пример настроек
├── .env                    # Ваши настройки (создать из .env.example)
├── input/                  # Входные файлы (для CLI-режима)
├── output/                 # Результаты (по категориям)
│   └── Маркетинг/
│       ├── *_speakers.txt
│       ├── *_summary.md
│       ├── *_actions.json
│       └── intermediate/   # Промежуточные чанки LLM (для отладки)
├── processed/              # Обработанные файлы (CLI)
├── failed/                 # Файлы с ошибками (CLI)
├── logs/                   # Лог-файлы (CLI)
└── scripts/
    ├── entrypoint.sh       # Точка входа контейнера (WhisperX)
    ├── postprocess.py      # Создание *_speakers.txt
    ├── summarize.py        # LLM-анализ (evidence-first pipeline)
    └── anonymize.py        # Локальная анонимизация PII
```

---

## Модели WhisperX

| Модель | VRAM | Качество | Скорость |
|---|---|---|---|
| `tiny` | ~1 GB | низкое | очень быстро |
| `base` | ~1 GB | среднее | быстро |
| `small` | ~2 GB | хорошее | быстро |
| `medium` | ~5 GB | очень хорошее | средне |
| `large-v2` | ~10 GB | отличное | медленно |
| `large-v3` | ~10 GB | лучшее | медленно |

Для CPU рекомендуется `small` или `medium`.

---

## Устранение неполадок

**Docker не запущен:**
```
[ERROR] Docker не запущен или не установлен.
```
→ Запустите Docker Desktop.

**HF_TOKEN не задан:**
```
[ERROR] HF_TOKEN не задан или содержит placeholder.
```
→ Укажите реальный токен в `.env`.

**Ошибка diarization (403 / unauthorized):**
→ Примите условия pyannote-моделей на huggingface.co (см. пункт 5 установки).

**Нет GPU / CUDA недоступен:**
→ Контейнер автоматически переключится на CPU. Укажите `DEVICE=cpu` и `COMPUTE_TYPE=int8` в `.env` явно.

**LLM-анализ не запускается:**
→ Проверьте, что `LLM_BASE_URL` задан в `.env` и LLM-сервер запущен.

**Файл с пробелами в имени (CLI):**
→ Оборачивайте путь в кавычки: `.\transcribe.ps1 ".\input\my meeting.mp3"`

**Порт 8080 занят:**
→ Измените в `docker-compose.yml`: `"8081:8080"` и откройте `http://localhost:8081`.


---

## Что делает проект

Вы кладёте аудио/видео файл в папку `input/`, запускаете скрипт — в папке `output/` появляются:

| Файл | Описание |
|---|---|
| `meeting.txt` | Сплошной текст транскрипта |
| `meeting.srt` | Субтитры |
| `meeting.json` | Полный JSON с временными метками и спикерами |
| `meeting.vtt` / `meeting.tsv` | Доп. форматы (если созданы WhisperX) |
| `meeting_speakers.txt` | Текст, сгруппированный по спикерам |

Пример `meeting_speakers.txt`:

```
[00:00:01 - 00:00:08] SPEAKER_00:
Всем привет, давайте начнём.

[00:00:09 - 00:00:20] SPEAKER_01:
Я обновлю статус по логистике.
```

---

## Требования

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows, macOS, Linux)
- PowerShell 5.1+ (встроен в Windows)
- Токен Hugging Face (бесплатно)
- Опционально: видеокарта NVIDIA с поддержкой CUDA

---

## Установка

### 1. Установите Docker Desktop

Скачайте и установите [Docker Desktop](https://www.docker.com/products/docker-desktop/).

### 2. Включите WSL2 backend (Windows)

При установке Docker Desktop включите опцию **"Use WSL 2 based engine"** (рекомендуется).  
Проверить: `Settings → General → Use the WSL 2 based engine`.

### 3. Проверьте GPU (опционально)

Если у вас видеокарта NVIDIA, проверьте поддержку GPU:

```powershell
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Если команда выводит информацию о GPU — всё готово. Если ошибка — проект будет работать на CPU (медленнее).

Для GPU требуется [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### 4. Получите токен Hugging Face

1. Зарегистрируйтесь на [huggingface.co](https://huggingface.co)
2. Перейдите в [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Создайте токен с правами `read`

### 5. Примите условия использования моделей pyannote

Войдите в Hugging Face и примите условия на страницах (кнопка **"Agree and access repository"**):

- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) — используется как PLDA-компонент внутри 3.1
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

> ⚠️ Нужно принять все три — иначе diarization упадёт с ошибкой 403.

### 6. Настройте `.env`

```powershell
cd meeting-transcriber
copy .env.example .env
```

Откройте `.env` и укажите свой токен:

```env
HF_TOKEN=hf_ваш_реальный_токен
MODEL=large-v3
LANGUAGE=ru
MIN_SPEAKERS=2
MAX_SPEAKERS=6
DEVICE=cuda
COMPUTE_TYPE=float16
```

#### CPU fallback (нет GPU)

Если GPU недоступен, измените в `.env`:

```env
DEVICE=cpu
COMPUTE_TYPE=int8
```

На CPU транскрибация займёт в 5–20 раз больше времени. Рекомендуется модель `medium` или `small`:

```env
MODEL=medium
```

---

## Запуск

### Один файл

```powershell
.\transcribe.ps1 .\input\meeting.mp3
```

### Batch-режим (все файлы из папки)

```powershell
.\transcribe.ps1 .\input
```

Поддерживаемые форматы: `.mp3`, `.wav`, `.m4a`, `.ogg`, `.mp4`, `.mkv`, `.flac`, `.aac`, `.webm`

---

## Первый запуск

При первом запуске Docker скачает и соберёт образ (~5–10 минут в зависимости от скорости интернета). Последующие запуски будут быстрее.

---

## Структура проекта

```
meeting-transcriber/
├── Dockerfile              # Docker-образ с WhisperX, PyTorch, ffmpeg
├── docker-compose.yml      # Compose-конфигурация
├── .env.example            # Пример настроек
├── .env                    # Ваши настройки (создать из .env.example)
├── README.md               # Эта документация
├── transcribe.ps1          # PowerShell-скрипт запуска
├── input/                  # Сюда кладите файлы для транскрибации
├── output/                 # Сюда сохраняются результаты
└── scripts/
    ├── entrypoint.sh       # Точка входа контейнера
    └── postprocess.py      # Постобработка: создание _speakers.txt
```

---

## Модели WhisperX

| Модель | VRAM | Качество | Скорость |
|---|---|---|---|
| `tiny` | ~1 GB | низкое | очень быстро |
| `base` | ~1 GB | среднее | быстро |
| `small` | ~2 GB | хорошее | быстро |
| `medium` | ~5 GB | очень хорошее | средне |
| `large-v2` | ~10 GB | отличное | медленно |
| `large-v3` | ~10 GB | лучшее | медленно |

Для CPU рекомендуется `small` или `medium`.

---

## Перенос на другой компьютер

1. Скопируйте папку `meeting-transcriber/` целиком
2. Создайте `.env` из `.env.example` на новом компьютере
3. Запустите — Docker образ соберётся автоматически

---

## Устранение неполадок

**Docker не запущен:**
```
[ERROR] Docker не запущен или не установлен.
```
→ Запустите Docker Desktop.

**HF_TOKEN не задан:**
```
[ERROR] HF_TOKEN не задан или содержит placeholder.
```
→ Укажите реальный токен в `.env`.

**Ошибка diarization (403 / unauthorized):**
→ Примите условия pyannote-моделей на huggingface.co (см. пункт 5 установки).

**Нет GPU / CUDA недоступен:**
→ Скрипт автоматически переключится на CPU. Укажите `DEVICE=cpu` и `COMPUTE_TYPE=int8` в `.env` явно.

**Файл с пробелами в имени:**
→ Оборачивайте путь в кавычки: `.\transcribe.ps1 ".\input\my meeting.mp3"`
