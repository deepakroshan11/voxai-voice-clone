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

# Install in strict order:
# 1. numpy first (runtime + build dep)
# 2. spacy-pkuseg — the maintained Python 3.11-compatible fork of pkuseg
#    (chatterbox-tts pulls in pkuseg==0.0.25 which fails to compile on Py3.11
#     due to removed longintrepr.h; spacy-pkuseg satisfies the same interface)
# 3. Everything else
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy>=1.24.0,<2.0.0" && \
    pip install --no-cache-dir "spacy-pkuseg" && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p outputs profiles .cache/huggingface && \
    chmod -R 777 outputs profiles .cache

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]