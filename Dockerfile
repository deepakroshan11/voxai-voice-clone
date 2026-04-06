FROM python:3.11-slim

# Install system deps including ffmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs on port 7860
ENV PORT=7860
ENV HOST=0.0.0.0
# Cache HuggingFace model weights inside the container
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

WORKDIR /app

# Copy requirements first (layer cache — only reinstalls if requirements.txt changes)
COPY backend/requirements.txt ./requirements.txt

# Install Python deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Create runtime directories
RUN mkdir -p outputs profiles .cache/huggingface

# HF Spaces requires port 7860
EXPOSE 7860

# Start server — running from /app so backend.main resolves correctly
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]