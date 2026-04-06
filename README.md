---
title: VOXAI Voice Clone Studio
emoji: 🎙️
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
app_port: 7860
---

# 🎙️ VOXAI — Voice Clone Studio

> **Sound Like Anyone — Instantly.**
> Upload a voice sample, type your text, and hear it reborn in seconds using state-of-the-art neural voice cloning.

![VOXAI Banner](https://img.shields.io/badge/VOXAI-Voice%20Clone%20Studio-cc2200?style=for-the-badge&logo=soundcloud&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)
![ChatterboxTTS](https://img.shields.io/badge/ChatterboxTTS-MIT-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## ✨ Features

- **Zero-shot voice cloning** — clone any voice from a 15–30 second audio sample
- **Live recording** — record your reference voice directly in the browser (no file needed)
- **Voice profiles** — save and reuse voices across sessions
- **7-stage professional audio chain** — bass restoration, cepstral de-noising, bandwidth extension, multiband dynamics, room tone, EBU R128 loudness normalization, true-peak limiting
- **48kHz broadcast-quality output** — upsampled from model's 24kHz via harmonic bandwidth extension
- **Multi-format input** — WAV, MP3, OGG, M4A, FLAC, WebM all supported
- **Retro vinyl UI** — animated splash screen, real-time waveform visualizer, vinyl disc recorder
- **Language-ready** — English available; Tamil, Hindi, Telugu, Spanish, French, Japanese coming soon

---

## 🏗️ Architecture

\`\`\`
voice-clone/
├── backend/
│   ├── main.py              # FastAPI app — v8 Definitive Fix
│   └── requirements.txt     # Python dependencies
├── frontend/
│   └── index.html           # Single-file vanilla JS UI
├── outputs/                 # Generated audio (gitignored)
├── profiles/                # Saved voice profiles (gitignored)
├── Dockerfile               # HuggingFace Spaces deployment
├── run.sh                   # One-command local startup
├── setup.sh                 # Environment setup script
└── README.md
\`\`\`

### Tech Stack

| Layer | Technology |
|-------|-----------|
| TTS Engine | [ChatterboxTTS](https://github.com/resemble-ai/chatterbox) (MIT, local, no API key) |
| Backend | FastAPI 0.111 + Uvicorn |
| Audio DSP | SciPy, torchaudio, NumPy |
| Frontend | Vanilla JS + Canvas API |
| Model Device | CUDA (auto) / CPU fallback |

---

## 🚀 Local Setup

### Prerequisites

- Python 3.10 or 3.11
- `ffmpeg` in PATH (for MP3/M4A/WebM uploads)
- ~2GB disk space for model weights (downloaded once on first run)
- GPU recommended (CUDA), CPU also works

### Install ffmpeg

\`\`\`bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows — download from https://www.gyan.dev/ffmpeg/builds/
# unzip and add the bin/ folder to your PATH
\`\`\`

### Setup & Run

\`\`\`bash
# Clone the repo
git clone https://github.com/deepakroshan11/voice-clone.git
cd voice-clone

# Run setup (creates venv + installs deps)
bash setup.sh

# Start the server
bash run.sh
\`\`\`

Open **http://localhost:8000** in your browser.

> **First run:** ChatterboxTTS will download ~2GB of model weights from HuggingFace. One-time download — subsequent runs start in seconds.

---

## 🌐 Deployment

### ✅ Recommended: Hugging Face Spaces (Free, Docker)

\`\`\`bash
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/voxai
git push hf main
\`\`\`

> 💡 Request a **ZeroGPU** grant on HF for free A100 GPU time — reduces clone time from ~120s to ~10s.

---

## 📡 API Reference

### `POST /clone`

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Text to synthesize (max 500 chars) |
| `voice_sample` | file | WAV/MP3/OGG audio (15–30s recommended) |
| `profile_name` | string | Use a saved profile instead of uploading |
| `language` | string | `en` (default) |
| `speed` | float | 0.5–2.0, default 1.0 |
| `exaggeration` | float | 0.1–1.5, default 0.3 |
| `cfg_weight` | float | 0.0–1.0, default 0.7 |

**Returns:** `audio/wav` at 48kHz

### `GET /health`

\`\`\`json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "ffmpeg": true,
  "sample_rate": 24000,
  "output_sr": 48000,
  "version": "v8-definitive"
}
\`\`\`

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| `Cannot convert '.mp3'` error | Install ffmpeg and add to PATH |
| `Model loading — please wait` | Wait ~60s after startup |
| Out of memory (CUDA) | Set `DEVICE = "cpu"` in main.py |
| Reference too short | Upload at least 4 seconds (15–30s recommended) |
| Audio sounds robotic | Quiet room, no music, 15–30s clean recording |

---

## 📄 License

MIT — free for personal and commercial use.

---

## 👤 Author

**Deepak Roshan** · AI Engineer
[deepakroshan380@gmail.com](mailto:deepakroshan380@gmail.com) · [GitHub @deepakroshan11](https://github.com/deepakroshan11)

---

*VOXAI · Voice Clone Studio · Est. 2025*