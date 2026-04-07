---
title: VOXAI Voice Clone Studio
emoji: 🎙️
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
app_port: 7860
---

```
██╗   ██╗ ██████╗ ██╗  ██╗ █████╗ ██╗
██║   ██║██╔═══██╗╚██╗██╔╝██╔══██╗██║
██║   ██║██║   ██║ ╚███╔╝ ███████║██║
╚██╗ ██╔╝██║   ██║ ██╔██╗ ██╔══██║██║
 ╚████╔╝ ╚██████╔╝██╔╝ ██╗██║  ██║██║
  ╚═══╝   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝
```

# 🎙️ VOXAI — Voice Clone Studio

> **Sound Like Anyone — Instantly.**
> Upload a voice sample, type your text, and hear it reborn in seconds using state-of-the-art neural voice cloning.

**Live Demo → [https://huggingface.co/spaces/deepakroshan/voxai](https://huggingface.co/spaces/deepakroshan/voxai)**
**GitHub → [https://github.com/deepakroshan11/voxai-voice-clone](https://github.com/deepakroshan11/voxai-voice-clone)**

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)
![ChatterboxTTS](https://img.shields.io/badge/Model-ChatterboxTTS-8B5CF6?style=flat-square)
![HuggingFace](https://img.shields.io/badge/Live_on-HuggingFace_Spaces-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Windows%20|%20Linux%20|%20macOS-lightgray?style=flat-square)

</div>

---

## What is VOXAI?

**VOXAI** is a fully local, zero-shot voice cloning web app. Upload or record a short audio sample of any voice, type your text, and get a synthesised audio clip that sounds like that person — all running on your own machine with no internet, no API key, and no subscription.

Powered by **[ChatterboxTTS](https://github.com/resemble-ai/chatterbox)** (Resemble AI, MIT licensed) — accent-preserving, CPU-friendly, no cloud dependency.

---

## ✨ Features

- 🎙️ **Live browser recording** — record your reference voice directly, no extra software
- 📁 **Multi-format upload** — WAV, MP3, OGG, M4A, FLAC, WebM all supported
- 👤 **Voice profiles** — save and reuse voices across sessions
- 🌍 **Accent preservation** — better Tamil / Indian English fidelity vs XTTS v2
- ⚡ **CPU + GPU** — works on any machine (GPU ~10× faster)
- 🎛️ **Expressiveness controls** — Exaggeration & CFG weight sliders
- 🔊 **7-stage professional audio chain** — bass restoration → cepstral de-noising → bandwidth extension → multiband dynamics → room tone → EBU R128 loudness normalisation → true-peak limiting
- 📡 **48kHz broadcast-quality output** — upsampled from model's 24kHz via harmonic bandwidth extension
- 🔒 **100% local** — nothing leaves your machine

---

## 📁 Project Structure

```
voxai-voice-clone/
├── backend/
│   ├── main.py              ← FastAPI app (Chatterbox TTS + 7-stage DSP)
│   └── requirements.txt     ← Python dependencies
├── frontend/
│   └── index.html           ← Single-file vanilla JS UI (retro vinyl theme)
├── outputs/                 ← Generated audio (gitignored)
├── profiles/                ← Saved voice profiles (gitignored)
├── Dockerfile               ← HuggingFace Spaces deployment
├── run.sh                   ← One-command local startup
├── setup.sh                 ← Environment setup script
└── README.md
```

---

## 🏗️ Architecture

```
Browser mic / file upload
        │
        ▼
  WAV conversion (browser-side for recordings, ffmpeg for uploads)
        │
        ▼
  Audio enhancement (high-pass filter → silence trim → normalise)
        │
        ▼
  ChatterboxTTS  ←  your text  +  exaggeration / cfg_weight
        │
        ▼
  7-stage DSP chain:
    1. Bass restoration
    2. Cepstral de-noising
    3. Bandwidth extension (24kHz → 48kHz)
    4. Multiband dynamics
    5. Room tone
    6. EBU R128 loudness normalisation
    7. True-peak limiter
        │
        ▼
  48kHz broadcast-quality WAV → streamed to browser
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| TTS Engine | ChatterboxTTS (MIT, local, no API key) |
| Backend | FastAPI 0.111 + Uvicorn |
| Audio DSP | SciPy, torchaudio, NumPy |
| Frontend | Vanilla JS + Canvas API (retro vinyl UI) |
| Model device | CUDA (auto) / CPU fallback |
| Cloud deploy | HuggingFace Spaces (Docker) |

---

## 🚀 Quick Start (Local)

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10 or 3.11 |
| RAM | 8 GB minimum (16 GB recommended) |
| Storage | ~3 GB (model weights + venv) |
| ffmpeg | Optional — needed for MP3/M4A/WebM uploads |

> **GPU (NVIDIA CUDA)** is optional but gives ~10× speed boost.

### Install ffmpeg

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows — download from https://www.gyan.dev/ffmpeg/builds/
# unzip and add the bin/ folder to PATH
```

### Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/deepakroshan11/voxai-voice-clone.git
cd voxai-voice-clone

# 2. Run setup (creates venv + installs deps)
bash setup.sh

# 3. Start the server
bash run.sh
```

Open **http://localhost:8000** in your browser.

> **First run:** ChatterboxTTS downloads ~2 GB of model weights from HuggingFace. One-time only — subsequent runs start in seconds.

### Manual Setup (Windows PowerShell)

```powershell
cd voxai-voice-clone

python -m venv venv
venv\Scripts\activate

pip install -r backend\requirements.txt

cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🌐 Cloud Deployment

### HuggingFace Spaces (Free, Docker)

The repo includes a ready-to-use `Dockerfile`. Push to your HF Space:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/voxai
git push hf main
```

> 💡 Request a **ZeroGPU** grant on HF for free A100 GPU time — reduces clone time from ~120s to ~10s on CPU.

---

## 📡 API Reference

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "ffmpeg": true,
  "sample_rate": 24000,
  "output_sr": 48000,
  "version": "v8-definitive"
}
```

### `POST /clone`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | required | Text to synthesise (max 500 chars) |
| `voice_sample` | file | — | WAV/MP3/OGG/M4A audio (15–30s recommended) |
| `profile_name` | string | — | Use a saved profile instead of uploading |
| `language` | string | `en` | Language code |
| `speed` | float | `1.0` | Playback speed (0.5–2.0) |
| `exaggeration` | float | `0.5` | 0 = flat, 1 = very expressive |
| `cfg_weight` | float | `0.5` | 0 = sounds like you, 1 = follows text closely |

**Returns:** `audio/wav` at 48kHz

### Other Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/save-profile` | Save a voice profile |
| `GET` | `/profiles` | List saved profiles |
| `DELETE` | `/profiles/{name}` | Delete a profile |
| `GET` | `/docs` | Interactive Swagger API docs |

---

## 🎯 Tips for Best Quality

1. **Record 15–20 seconds** of natural speech — biggest single improvement
2. **Quiet room** — background noise significantly hurts quality
3. **Speak naturally** at your normal pace, not slowly or carefully
4. **Use WAV or M4A** from a phone voice memo for best mic quality
5. **Save as a profile** so you don't re-upload every time

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| `Cannot convert '.mp3'` error | Install ffmpeg and add to PATH |
| `Model loading — please wait` | Wait ~60s after first startup |
| Out of memory (CUDA) | Set `DEVICE = "cpu"` in `backend/main.py` |
| Reference too short | Upload at least 4 seconds (15–30s recommended) |
| Audio sounds robotic | Quiet room, no music, 15–30s clean recording |
| Port 8000 already in use | `lsof -ti:8000 \| xargs kill -9` |

---

## 📄 License

MIT — free for personal and commercial use.

---

## 👤 Author

**Deepak Roshan** · AI Engineer
[deepakroshan380@gmail.com](mailto:deepakroshan380@gmail.com) · [GitHub @deepakroshan11](https://github.com/deepakroshan11)

---

<div align="center">

*VOXAI · Voice Clone Studio · Est. 2025*

![VOXAI](https://img.shields.io/badge/VOXAI-Voice%20Clone%20Studio-cc2200?style=for-the-badge&logo=soundcloud&logoColor=white)

</div>
