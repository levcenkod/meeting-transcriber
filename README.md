# meeting-transcriber

Локальный Docker-проект для автоматической транскрибации аудио/видео записей встреч с разделением по спикерам (diarization) через **WhisperX**.

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
