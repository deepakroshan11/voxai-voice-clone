FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PORT=7860
ENV HOST=0.0.0.0
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt

# ── Critical install order ──────────────────────────────────────────
# 1. numpy must be in the GLOBAL site-packages before pkuseg builds,
#    because pkuseg's setup.py calls `import numpy` at build time.
#    pip's isolated build env does NOT inherit the parent env, so we
#    must pre-seed numpy into the build cache via PIP_NO_BUILD_ISOLATION=0
#    OR simply install numpy+pkuseg before everything else.
#
# 2. pkuseg must be installed (and compiled) before chatterbox-tts,
#    so it is already satisfied when chatterbox pulls it in.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy>=1.24.0,<2.0.0" && \
    pip install --no-cache-dir --no-build-isolation "pkuseg==0.0.25" && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p outputs profiles .cache/huggingface && \
    chmod -R 777 outputs profiles .cache

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]