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

# Create working directories
RUN mkdir -p /input /output /scripts

# Copy postprocess script
COPY scripts/postprocess.py /scripts/postprocess.py

# Copy summarize script
COPY scripts/summarize.py /scripts/summarize.py

# Copy entrypoint script
COPY scripts/entrypoint.sh /scripts/entrypoint.sh
RUN chmod +x /scripts/entrypoint.sh

WORKDIR /

ENTRYPOINT ["/scripts/entrypoint.sh"]
