<div align="center">

```
██╗   ██╗ ██████╗ ██╗  ██╗ █████╗ ██╗
██║   ██║██╔═══██╗╚██╗██╔╝██╔══██╗██║
██║   ██║██║   ██║ ╚███╔╝ ███████║██║
╚██╗ ██╔╝██║   ██║ ██╔██╗ ██╔══██║██║
 ╚████╔╝ ╚██████╔╝██╔╝ ██╗██║  ██║██║
  ╚═══╝   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝
```

**Zero-Shot Voice Cloning — Local, Free, No API Key**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi)
![Chatterbox TTS](https://img.shields.io/badge/Model-Chatterbox%20TTS-8B5CF6?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey?style=flat-square)

</div>

---

## What is VOXAI?

**VOXAI** is a fully local, zero-shot voice cloning web app. Upload or record a short audio sample of any voice, type your text, and get a synthesised audio clip that sounds like that person — all running on your own machine with no internet, no API key, and no subscription.

**Powered by [Chatterbox TTS](https://github.com/resemble-ai/chatterbox)** — MIT licensed, accent-preserving, CPU-friendly.

---

## Features

- 🎙️ **Live browser recording** — record directly in the browser, no extra software
- 📁 **File upload** — upload WAV, MP3, M4A, OGG, WebM
- 👤 **Voice profiles** — save voices and reuse them anytime
- 🌍 **Accent preservation** — better Tamil / Indian English accent fidelity vs XTTS v2
- ⚡ **CPU + GPU** — works on any machine (GPU gives ~10× speed boost)
- 🎛️ **Expressiveness controls** — Exaggeration & CFG weight sliders
- 🔒 **100% local** — nothing leaves your machine

---

## Requirements

| Requirement | Version |
|---|---|
| Python | **3.10 or 3.11** |
| RAM | 8 GB minimum (16 GB recommended) |
| Storage | ~3 GB (model weights + venv) |
| ffmpeg | Optional (needed for MP3/M4A uploads) |

> **GPU (NVIDIA CUDA)** is optional but recommended for faster generation.

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/voxai-voice-clone.git
cd voxai-voice-clone/voice-clone
```

### 2. Run setup (creates venv + installs dependencies)

```bash
bash setup.sh
```

> First run downloads ~2 GB of Chatterbox model weights from HuggingFace. This only happens once.

### 3. Start the server

```bash
bash run.sh
```

### 4. Open in browser

```
http://localhost:8000
```

---

## Manual Setup (Windows PowerShell)

```powershell
cd voice-clone

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r backend\requirements.txt

# Start server
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Project Structure

```
voice-clone/
├── backend/
│   ├── main.py              ← FastAPI backend (Chatterbox TTS)
│   ├── requirements.txt     ← Python dependencies
│   └── venv/                ← Virtual environment (not committed)
├── frontend/
│   └── index.html           ← Single-file web UI
├── outputs/                 ← Generated audio files (not committed)
├── profiles/                ← Saved voice profiles (not committed)
├── run.sh                   ← Start server
├── setup.sh                 ← One-time setup
└── README.md
```

---

## How It Works

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
  Chatterbox TTS  ←  your text
        │
        ▼
  Cloned audio (WAV) → streamed back to browser
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server & model status |
| `POST` | `/clone` | Generate cloned voice |
| `POST` | `/save-profile` | Save a voice profile |
| `GET` | `/profiles` | List saved profiles |
| `DELETE` | `/profiles/{name}` | Delete a profile |
| `GET` | `/docs` | Interactive API docs (Swagger) |

### POST /clone — Parameters

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string | required | Text to synthesise |
| `voice_sample` | file | — | Audio file (WAV/MP3/M4A/OGG) |
| `profile_name` | string | — | Use a saved profile instead |
| `exaggeration` | float | `0.5` | 0 = flat, 1 = very expressive |
| `cfg_weight` | float | `0.5` | 0 = sounds like you, 1 = follows text closely |
| `speed` | float | `1.0` | Playback speed (0.7–1.5) |

---

## Tips for Best Voice Clone Quality

1. **Record 15–20 seconds** of natural speech (biggest single improvement)
2. **Quiet room** — background noise hurts quality significantly
3. **Speak naturally** at your normal pace, not slowly or carefully
4. **Use WAV or M4A** from a phone voice memo for best mic quality
5. **Save as a profile** so you don't have to re-upload every time

---

## ffmpeg (optional)

ffmpeg is only needed if you upload **MP3, M4A, or WebM** files.
Browser recordings are converted to WAV automatically in the browser.

**Windows install:**
1. Download from [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/)
2. Unzip and copy the `bin/` folder path
3. Add it to System Environment Variables → PATH
4. Restart your terminal

---

## Tech Stack

| Component | Technology |
|---|---|
| Voice Model | [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) (Resemble AI) |
| Backend | FastAPI + Uvicorn |
| Audio processing | torchaudio, scipy, soundfile |
| Frontend | Vanilla JS + HTML/CSS (single file) |
| Browser audio | Web Audio API + MediaRecorder |

---

## Author

**Deepak Roshan**
📧 [deepakroshan380@gmail.com](mailto:deepakroshan380@gmail.com)

---

## License

MIT — free to use, modify, and distribute.

---

<div align="center">
Built with ❤️ by <a href="mailto:deepakroshan380@gmail.com">Deepak Roshan</a>
</div>