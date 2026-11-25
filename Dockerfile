FROM python:3.11-slim

# ── System deps ────────────────────────────────────────────────────────────────
# ffmpeg: reads uploads (mp3/m4a/webm...) and converts downloads (/export).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ── Hugging Face Spaces run the container as user 1000 ────────────────────────
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860 \
    HF_HOME=/home/user/.cache/huggingface \
    TORCH_HOME=/home/user/.cache/torch \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive
WORKDIR /home/user/app

# ── Python deps (same versions as the tested local venv) ──────────────────────
# 1. CPU-only PyTorch 2.6.0 (Spaces' free hardware has no GPU; ~2 GB smaller).
# 2. Chatterbox 0.1.7 = Turbo English engine (uses spacy-pkuseg, so the old
#    pkuseg build problem is gone).
# 3. IndicF5 (Tamil) from AI4Bharat, held to the same torch/numpy/transformers.
# 4. Web server + audio utilities.
COPY --chown=user constraints.txt backend/requirements.txt ./
RUN pip install --user --upgrade pip \
    && pip install --user --index-url https://download.pytorch.org/whl/cpu \
       "torch==2.6.0" "torchaudio==2.6.0" \
    && pip install --user -c constraints.txt "chatterbox-tts==0.1.7" \
    && pip install --user -c constraints.txt "git+https://github.com/ai4bharat/IndicF5.git" \
    && pip install --user -c constraints.txt -r requirements.txt

# ── App source ─────────────────────────────────────────────────────────────────
COPY --chown=user . .
RUN mkdir -p outputs profiles /home/user/.cache/huggingface /home/user/.cache/torch

# Models download on first start (~6 GB: Chatterbox-Turbo, IndicF5, Whisper,
# Demucs). IndicF5 is gated: accept its terms at huggingface.co/ai4bharat/IndicF5
# and add an HF_TOKEN secret in the Space settings.
EXPOSE 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
