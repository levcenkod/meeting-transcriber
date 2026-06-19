#!/bin/bash
# entrypoint.sh — запускает WhisperX с параметрами из переменных окружения

set -e

INPUT_FILE="$1"

if [ -z "$INPUT_FILE" ]; then
    echo "[ERROR] Не передан файл для транскрибации."
    echo "Использование: docker run ... meeting-transcriber /input/meeting.mp3"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "[ERROR] Файл не найден: $INPUT_FILE"
    exit 1
fi

# Проверка обязательных переменных
if [ -z "$HF_TOKEN" ]; then
    echo "[ERROR] HF_TOKEN не задан. Укажите токен Hugging Face в .env файле."
    exit 1
fi

# Параметры с defaults
MODEL="${MODEL:-large-v3}"
LANGUAGE="${LANGUAGE:-ru}"
DEVICE="${DEVICE:-cuda}"
COMPUTE_TYPE="${COMPUTE_TYPE:-float16}"
DIARIZE_MODEL="${DIARIZE_MODEL:-pyannote/speaker-diarization-3.1}"

# CPU fallback: если cuda недоступен, автоматически переключаемся на cpu
if [ "$DEVICE" = "cuda" ]; then
    python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null || {
        echo "[WARN] CUDA недоступен, переключаемся на CPU (int8)."
        DEVICE="cpu"
        COMPUTE_TYPE="int8"
    }
fi

echo "[INFO] Файл:         $INPUT_FILE"
echo "[INFO] Модель:       $MODEL"
echo "[INFO] Язык:         $LANGUAGE"
echo "[INFO] Устройство:   $DEVICE ($COMPUTE_TYPE)"
echo "[INFO] Модель диар.: $DIARIZE_MODEL"

# Папка для вывода (с поддержкой подкатегорий)
OUTPUT_DIR="/output"
if [ -n "${OUTPUT_SUBDIR:-}" ]; then
    OUTPUT_DIR="/output/$OUTPUT_SUBDIR"
    mkdir -p "$OUTPUT_DIR"
    echo "[INFO] Категория:    $OUTPUT_SUBDIR  ->  $OUTPUT_DIR"
fi

# Формируем аргументы для количества спикеров (опционально)
SPEAKER_ARGS=""
if [ -n "$MIN_SPEAKERS" ]; then
    SPEAKER_ARGS="$SPEAKER_ARGS --min_speakers $MIN_SPEAKERS"
fi
if [ -n "$MAX_SPEAKERS" ]; then
    SPEAKER_ARGS="$SPEAKER_ARGS --max_speakers $MAX_SPEAKERS"
fi

echo "[INFO] Запуск WhisperX..."

# NB: whisperx 3.1.1 не принимает --diarize_model в CLI — модель диаризации в нём
# зашита как pyannote/speaker-diarization-3.1 (совпадает с DIARIZE_MODEL по умолчанию).
# Передача --diarize_model вызывала "unrecognized arguments", поэтому здесь его нет.
whisperx "$INPUT_FILE" \
    --model "$MODEL" \
    --language "$LANGUAGE" \
    --diarize \
    --hf_token "$HF_TOKEN" \
    --device "$DEVICE" \
    --compute_type "$COMPUTE_TYPE" \
    --output_dir "$OUTPUT_DIR" \
    $SPEAKER_ARGS

echo "[INFO] WhisperX завершён. Результаты в $OUTPUT_DIR"