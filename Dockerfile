FROM python:3.11-slim

# ── System deps ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Environment ────────────────────────────────────────────────────────────────
ENV PORT=7860
ENV HOST=0.0.0.0
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt

# ── Python deps ────────────────────────────────────────────────────────────────
#
# ROOT CAUSE: pkuseg==0.0.25 uses longintrepr.h (removed from Python 3.11-slim)
# and also fails pip's build-isolation (no numpy in subprocess).
# It cannot be compiled on this image under any circumstances.
#
# SOLUTION:
#   1. Install torch/torchaudio at the version chatterbox requires FIRST.
#   2. Install all of chatterbox-tts's real runtime deps manually.
#   3. Install chatterbox-tts itself with --no-deps so pip never resolves pkuseg.
#   4. Install spacy-pkuseg as a runtime shim (same interface as pkuseg).
#   5. Install the rest of requirements.txt — chatterbox is already present so
#      pip won't re-resolve its deps and won't try to build pkuseg.
#
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
    "numpy>=1.24.0,<1.26.0" \
    "torch==2.6.0" \
    "torchaudio==2.6.0" \
    && pip install --no-cache-dir \
    "librosa==0.11.0" \
    "transformers==4.46.3" \
    "diffusers==0.29.0" \
    "safetensors==0.5.3" \
    "conformer==0.3.2" \
    "resemble-perth==1.0.1" \
    "s3tokenizer" \
    "pykakasi==2.3.0" \
    "gradio==5.44.1" \
    && pip install --no-cache-dir "spacy-pkuseg" \
    && pip install --no-cache-dir --no-deps "chatterbox-tts==0.1.4" \
    && pip install --no-cache-dir -r requirements.txt

# ── App source ─────────────────────────────────────────────────────────────────
COPY . .

RUN mkdir -p outputs profiles .cache/huggingface \
    && chmod -R 777 outputs profiles .cache

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]