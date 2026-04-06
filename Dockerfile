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
# Install in strict order to work around pkuseg==0.0.25 build-isolation issue:
#
# Step 1 — Upgrade pip
# Step 2 — Install numpy first (it is a build-time dep of pkuseg AND a runtime dep)
# Step 3 — Install pkuseg with --no-build-isolation so it can see numpy in the
#           outer environment instead of spinning up an isolated subprocess that
#           cannot find numpy. This resolves the "ModuleNotFoundError: No module
#           named 'numpy'" that kills the build on Python 3.11.
# Step 4 — Install spacy-pkuseg (the maintained Python 3.11 compatible fork).
#           chatterbox-tts declares `pkuseg` as a dep; pip will see pkuseg is
#           already satisfied and skip the failing source-build.
# Step 5 — Install everything else from requirements.txt.
#           --no-deps on chatterbox-tts is NOT needed because pkuseg is already
#           present, so pip won't re-download / re-build it.

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "numpy>=1.24.0,<2.0.0" \
    && pip install --no-cache-dir --no-build-isolation "pkuseg==0.0.25" \
    && pip install --no-cache-dir "spacy-pkuseg" \
    && pip install --no-cache-dir -r requirements.txt

# ── App source ─────────────────────────────────────────────────────────────────
COPY . .

RUN mkdir -p outputs profiles .cache/huggingface \
    && chmod -R 777 outputs profiles .cache

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]