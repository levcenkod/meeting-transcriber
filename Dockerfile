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
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support (also works on CPU)
RUN pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Pin torch/torchaudio/triton so that the pip resolver never re-downloads them
# while installing whisperx and its deps (was causing 20-min hangs due to
# backtracking over 2 GB CUDA wheels).
# numpy is held at <2 because some pyannote internals still use np.NaN (removed
# in NumPy 2.0). numpy 1.26.4 works fine with everything below.
#
# We do NOT pin faster-whisper / transformers / huggingface-hub here.
# whisperx 3.1.1 (PyPI) declares faster-whisper>=0.10.0 in its own setup.py,
# so pinning 0.9.0 caused ResolutionImpossible. Instead we use the latest
# official whisperx (3.8+) which is written for modern faster-whisper and
# resolves cleanly. The old --diarize_model arg has already been removed from
# entrypoint.sh so there is no CLI mismatch.
RUN printf 'torch==2.1.2\ntorchaudio==2.1.2\ntriton==2.1.0\nnumpy<2\n' > /constraints.txt
ENV PIP_CONSTRAINT=/constraints.txt

# Install WhisperX (latest official release), OpenAI client, Flask and spaCy.
# The constraint above keeps torch/torchaudio/triton fixed across all installs.
RUN pip install whisperx openai flask spacy

# Download spaCy models for local PII anonymization (used by anonymize.py).
# Models downloaded at build time; falls back to regex-only if download fails
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
