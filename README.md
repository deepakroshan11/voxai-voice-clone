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

```
voice-clone/
├── backend/
│   ├── main.py              # FastAPI app — v8 Definitive Fix
│   └── requirements.txt     # Python dependencies
├── frontend/
│   └── index.html           # Single-file vanilla JS UI
├── outputs/                 # Generated audio (gitignored)
├── profiles/                # Saved voice profiles (gitignored)
├── run.sh                   # One-command startup
├── setup.sh                 # Environment setup script
└── README.md
```

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

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows — download from https://www.gyan.dev/ffmpeg/builds/
# unzip and add the bin/ folder to your PATH
```

### Setup & Run

```bash
# Clone the repo
git clone https://github.com/deepakroshan11/voice-clone.git
cd voice-clone

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Start the server
bash run.sh
# OR manually:
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser.

> **First run:** ChatterboxTTS will download ~2GB of model weights from HuggingFace. This is a one-time download — subsequent runs start in seconds.

---

## 🌐 Deployment

### ✅ Recommended: Hugging Face Spaces (Free, Docker)

Hugging Face Spaces offers free CPU and GPU instances that can handle the ~2GB ChatterboxTTS model. This is the best free-tier option.

**See [`Dockerfile`](./Dockerfile) and [`app.py`](./app.py) in the repo root for the HF Spaces deployment.**

Steps:
1. Create a Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Docker** as the SDK
3. Push this repo — the `Dockerfile` handles everything

> 💡 Request a **ZeroGPU** grant on HF for free A100 GPU time — greatly speeds up cloning.

---

## 📡 API Reference

### `POST /clone`

Clone a voice.

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Text to synthesize (max 500 chars) |
| `voice_sample` | file | WAV/MP3/OGG audio (15–30s recommended) |
| `profile_name` | string | Use a saved profile instead of uploading |
| `language` | string | `en` (default) |
| `speed` | float | 0.5–2.0, default 1.0 |
| `exaggeration` | float | 0.1–1.5, default 0.3 |
| `cfg_weight` | float | 0.0–1.0, default 0.7 |

**Returns:** `audio/wav` file at 48kHz

---

### `POST /save-profile`

Save a voice for reuse.

| Field | Type | Description |
|-------|------|-------------|
| `profile_name` | string | Name for the profile |
| `voice_sample` | file | Reference audio |

---

### `GET /profiles`

List saved voice profiles.

### `DELETE /profiles/{name}`

Delete a voice profile.

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

---

## 🎨 UI Highlights

- **Animated vinyl splash screen** with real-time waveform and loading progress
- **Tab-based input** — drag-and-drop upload OR live disc recorder with waveform
- **Inline language selector** with animated expand/collapse
- **Real-time audio visualizer** on playback
- **Toast notifications** for all state changes
- Retro aesthetic: VT323 font, cream/ink palette, scanline overlays, custom cursor

---

## ⚙️ Configuration

Key constants in `backend/main.py`:

```python
CHATTERBOX_SR        = 24000   # model native sample rate
OUTPUT_SR            = 48000   # broadcast output sample rate
DEFAULT_EXAGGERATION = 0.3     # pitch stability (lower = more stable)
DEFAULT_CFG_WEIGHT   = 0.7     # classifier-free guidance
BEST_DURATION_S      = 15      # optimal reference audio length
CHUNK_CHAR_LIMIT     = 130     # max chars per synthesis chunk
```

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| `Cannot convert '.mp3'` error | Install ffmpeg and add to PATH |
| `Model loading — please wait` on `/clone` | Wait ~60s after startup for model to load |
| Out of memory (CUDA) | Switch to CPU: set `DEVICE = "cpu"` in main.py, or reduce text length |
| Reference too short error | Record/upload at least 4 seconds (15–30s recommended) |
| Audio sounds robotic | Use a clean recording (quiet room, no music), 15–30s, avoid WhatsApp compressed audio |

---

## 📄 License

MIT — free for personal and commercial use. ChatterboxTTS is also MIT licensed.

---

## 👤 Author

**Deepak Roshan**
AI Engineer
[deepakroshan380@gmail.com](mailto:deepakroshan380@gmail.com) · [GitHub @deepakroshan11](https://github.com/deepakroshan11)

---

*VOXAI · Voice Clone Studio · Est. 2025*