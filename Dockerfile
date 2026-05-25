# Base image with CUDA support (works for both GPU and CPU)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support (also works on CPU)
RUN pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install WhisperX and dependencies
RUN pip install whisperx

# Install pyannote.audio for diarization
RUN pip install pyannote.audio

# Install OpenAI-compatible client (used by summarize.py)
RUN pip install openai

# Install Flask for the web UI
RUN pip install flask

# Install spaCy for local PII anonymization (used by anonymize.py)
# Models downloaded at build time; falls back to regex-only if download fails
RUN pip install spacy
RUN python -m spacy download ru_core_news_md || python -m spacy download ru_core_news_sm || echo "[WARN] No Russian spaCy model - anonymization will use regex only"
RUN python -m spacy download en_core_web_sm || echo "[WARN] No English spaCy model"

# Create working directories
RUN mkdir -p /input /output /scripts

# Copy postprocess script
COPY scripts/postprocess.py /scripts/postprocess.py

# Copy anonymize module
COPY scripts/anonymize.py /scripts/anonymize.py

# Copy summarize script
COPY scripts/summarize.py /scripts/summarize.py

# Copy entrypoint script
COPY scripts/entrypoint.sh /scripts/entrypoint.sh
RUN chmod +x /scripts/entrypoint.sh

# Web UI
RUN mkdir -p /app/templates /app/static
COPY app.py /app/app.py
COPY templates/index.html /app/templates/index.html
COPY templates/settings.html /app/templates/settings.html
COPY static/ /app/static/

WORKDIR /

ENTRYPOINT ["/scripts/entrypoint.sh"]
