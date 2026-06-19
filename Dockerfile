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

# Pin torch/torchaudio (and triton — torch's CUDA dependency) via a pip
# constraints file so that NO subsequent install (whisperx, pyannote.audio, ...)
# is ever allowed to swap them out.
# Without this, pip's dependency resolver "backtracks" and re-downloads multiple
# 2+ GB torch wheels and the ~200 MB triton wheel (and may replace the CUDA build
# with a CPU build), which is what makes the build hang for ~20 minutes.
# numpy is pinned to the last 1.x release: pyannote.audio still uses np.NaN,
# which was removed in NumPy 2.0 (would crash WhisperX at runtime).
RUN printf 'torch==2.1.2\ntorchaudio==2.1.2\ntriton==2.1.0\nnumpy<2\n' > /constraints.txt
ENV PIP_CONSTRAINT=/constraints.txt

# Install WhisperX (pinned), OpenAI client, Flask and spaCy in a single
# resolver pass. The constraint above keeps torch fixed.
#
# WhisperX is pinned to 3.1.1 on purpose:
#   * its CLI matches entrypoint.sh (--diarize_model, --hf_token, --compute_type).
#     Newer whisperx releases got dragged in/out by the resolver and dropped
#     --diarize_model, causing "unrecognized arguments: --diarize_model".
#   * it pulls pyannote.audio==3.1.1 and a ctranslate2/faster-whisper combo that
#     is compatible with this image's CUDA 12.1 + cuDNN 8 (newer ones need cuDNN 9).
RUN pip install whisperx==3.1.1 openai flask spacy

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
