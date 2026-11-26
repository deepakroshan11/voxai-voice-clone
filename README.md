---
title: VOXAI Voice Clone Studio
emoji: 🎙️
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
app_port: 7860
startup_duration_timeout: 1h
---

<div align="center">

# VOXAI — Voice Clone Studio

**Sound like anyone, in English and தமிழ்.**

Upload or record a short voice sample, type your text, and get natural speech in that voice.

[**Open the app**](https://deepakroshan-voxai.hf.space) · [Hugging Face Space](https://huggingface.co/spaces/deepakroshan/voxai) · [Source on GitHub](https://github.com/deepakroshan11/voxai-voice-clone)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.129-009688?style=flat-square&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Languages](https://img.shields.io/badge/Languages-English%20%7C%20Tamil-cc2200?style=flat-square)
![Deploy](https://img.shields.io/badge/Deploy-Docker%20%7C%20HF%20Spaces-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## Overview

VOXAI is a zero-shot voice cloning web application. A few seconds of someone's voice is enough: no training, no account, no API key. It runs on your own machine or as a Docker container, and is live on Hugging Face Spaces.

| Language | Engine | Notes |
|---|---|---|
| English | [Chatterbox-Turbo](https://github.com/resemble-ai/chatterbox) (Resemble AI) | Fast; supports expressive sound tags |
| Tamil (தமிழ்) | [IndicF5](https://huggingface.co/ai4bharat/IndicF5) (AI4Bharat) | Natural spoken Tamil by default |

## Features

**Voice cloning**
- Clone any voice from a 5–30 second sample, uploaded (WAV, MP3, M4A, OGG, FLAC, WebM) or recorded in the browser
- English and Tamil from the same voice sample
- Saved voice profiles: preview a voice and reuse it without uploading again

**Natural Tamil**
- Converts written Tamil into everyday spoken Tamil (இருக்கிறது → இருக்கு, போகிறேன் → போறேன்), with a formal option
- Numbers, currency (₹) and percentages are spoken as Tamil words
- The words in the voice sample are detected automatically (Whisper), so no transcript has to be typed and no words are dropped

**Expressive English**
- Sound tags placed between words: `[laugh]` `[chuckle]` `[sigh]` `[gasp]` `[clear throat]` `[whispering]` `[happy]` `[sarcastic]`

**Studio-grade voice handling**
- Voice isolation: background music and noise are removed from the sample before cloning (Hybrid Demucs)
- The cleanest stretch of a long sample is used, always cut at a pause
- Pauses and speaking pace follow the speaker's own rhythm, measured from the sample
- Clean 48 kHz output at a consistent loudness (−16 LUFS)

**Export**
- Download as MP3, WAV, FLAC, M4A, OGG, OPUS, AIFF or WMA, with your own file name

**Voice-to-voice (API)**
- Record yourself reading a line; it comes back in the cloned voice with your exact timing and pauses (`POST /perform`)

## How it works

```
Voice sample ──► convert ──► isolate voice ──► pick clean, lively window ──► measure rhythm + melody range
                                                    │
Text ──► (Tamil: spoken-Tamil rewrite, numbers)     │
                                                    ▼
                           English: Chatterbox-Turbo │ Tamil: IndicF5 (+ auto transcript)
                                                    │
                                                    ▼
   join with the speaker's pauses ──► match the speaker's intonation ──► 48 kHz, −16 LUFS ──► WAV / export
```

**Intonation matching.** Cloned speech, especially Tamil in male voices, often comes out flatter than the speaker: its pitch melody moves about 1 semitone where real speech moves 3–5. VOXAI measures the speaker's own melody range on the voice sample. If the clone is flatter, it widens only the slow phrase melody (rises, falls and stressed words) back toward that range, using PSOLA resynthesis, which keeps the voice's timbre and timing. The small natural wobble of the voice is kept as it is, and clones that are already lively are left untouched.

## Tech stack

| Layer | Technology |
|---|---|
| Speech models | Chatterbox-Turbo, IndicF5, Whisper (language ID + Tamil ASR), Hybrid Demucs |
| Backend | FastAPI, Uvicorn, PyTorch 2.6 (CPU or CUDA) |
| Audio | torchaudio, librosa, SciPy, pyloudnorm, ffmpeg |
| Frontend | Single-page HTML / CSS / JavaScript, no build step |
| Deployment | Docker, Hugging Face Spaces |

## Getting started

### Requirements

| | Minimum | Recommended |
|---|---|---|
| Python | 3.11 | 3.11 |
| RAM | 8 GB | 16 GB |
| Disk | 10 GB (model weights) | 15 GB |
| GPU | Not required | NVIDIA GPU with CUDA for fast Tamil |
| ffmpeg | Required | — |

Tamil uses IndicF5, a gated model: accept its terms at [huggingface.co/ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5) and provide a Hugging Face token (`HF_TOKEN`, or `hf auth login`).

### Run with Docker (recommended)

```bash
git clone https://github.com/deepakroshan11/voxai-voice-clone.git
cd voxai-voice-clone
docker build -t voxai .
docker run -p 7860:7860 -e HF_TOKEN=<your-token> voxai
```

Open **http://localhost:7860**. The first start downloads about 6 GB of model weights.

### Run locally

```bash
git clone https://github.com/deepakroshan11/voxai-voice-clone.git
cd voxai-voice-clone
bash setup.sh      # virtual environment + pinned dependencies
bash run.sh        # http://localhost:8000
```

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install "torch==2.6.0" "torchaudio==2.6.0"
pip install -c constraints.txt "chatterbox-tts==0.1.7"
pip install -c constraints.txt "git+https://github.com/ai4bharat/IndicF5.git"
pip install -c constraints.txt -r backend\requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

`constraints.txt` pins the versions that are tested together (PyTorch, NumPy, Transformers). Install with it, and do not upgrade those packages separately.

## Deploying to Hugging Face Spaces

1. Create a Docker Space and push this repository to it.
2. Under **Settings → Variables and secrets**, add the secret `HF_TOKEN` (with access to IndicF5).
3. Optional: add the variable `VOXAI_LOG_REPO` (see [Data and privacy](#data-and-privacy)).

The free CPU tier runs both languages; a GPU tier makes Tamil generation much faster.

On a Space, VOXAI switches on its public-site protections automatically:

- **Private voices.** Each browser's saved profiles are stored separately, so visitors never see, play, use or delete each other's voices.
- **Safe uploads.** Only real audio files are accepted (WAV, MP3, M4A, OGG, FLAC, WebM, up to 40 MB), and ffmpeg reads them with a fixed format and no access to other files or the network.
- **Fair use.** One job runs at a time, and the page shows each visitor their place in line. Each visitor is limited to 5 clones per 10 minutes.
- **Consent.** Each clone needs the "I own this voice or have permission" box ticked.
- **Locked-down API.** Other websites cannot call the API from their pages.

Give `HF_TOKEN` the smallest access that works: read access to `ai4bharat/IndicF5`, plus write access to the activity-log dataset if you use one.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | — | Hugging Face token; required for the gated Tamil model |
| `VOXAI_EN_ENGINE` | `turbo` | English engine: `turbo` or `standard` (original Chatterbox) |
| `VOXAI_INDIC_STEPS` | `32` | Tamil quality/speed: 32 = best, 16 ≈ 2× faster |
| `VOXAI_PACE` | `0.9` | Tamil reading pace relative to the speaker's own rate |
| `VOXAI_ISOLATE` | `auto` | Voice isolation: `auto` (when the sample has background), `on`, `off` |
| `VOXAI_INTONATION` | `on` | Match the speaker's intonation range (`off` to disable) |
| `VOXAI_THREADS` | `0` | CPU threads for PyTorch (0 = automatic) |
| `PRELOAD_INDIC` | `1` | Load the Tamil model at startup |
| `VOXAI_LOG_REPO` | — | Private dataset for the activity log (off when empty) |
| `VOXAI_PRIVATE_PROFILES` | on for Spaces | Keep each browser's saved voices separate |
| `VOXAI_REQUIRE_CONSENT` | on for Spaces | Require `consent=1` on `/clone` and `/perform` |
| `VOXAI_RATE_LIMIT` | `5/600` on Spaces, off locally | Clones per visitor per time window (seconds); `0/1` turns it off |
| `VOXAI_MAX_JOBS` | `1` | Jobs that run at the same time |
| `VOXAI_MAX_WAITING` | `8` | Jobs that may wait in line before new ones are turned away |
| `VOXAI_MAX_UPLOAD_MB` | `40` | Largest accepted upload |
| `VOXAI_MAX_PROFILES` | `12` | Saved voices per browser (private profiles only) |
| `VOXAI_ORIGINS` | the Space URL + localhost | Comma-separated websites allowed to call the API from a browser |
| `PORT` | `8000` | Port when started with `python backend/main.py` |

## API

Interactive documentation is served at `/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/clone` | Text to speech in the given voice. Returns 48 kHz WAV |
| `POST` | `/perform` | Voice-to-voice: a spoken recording, converted to the given voice |
| `POST` | `/transcribe` | Words spoken in a voice sample (language detected automatically) |
| `POST` | `/export` | Convert a generated WAV to `mp3`, `wav`, `flac`, `m4a`, `ogg`, `opus`, `aiff` or `wma` |
| `POST` | `/save-profile` | Save a voice profile |
| `GET` | `/profiles` | List saved profiles |
| `GET` | `/profiles/{name}/audio` | The saved voice sample of a profile |
| `DELETE` | `/profiles/{name}` | Delete a profile |
| `GET` | `/queue/{ticket}` | Place in line of a job sent with that `ticket` |
| `GET` | `/health` | Status of the models, ffmpeg and available formats |

With private profiles on (the default on Spaces), the profile endpoints, and `profile_name` in other requests, need an `X-Client-Id` header: a random id of 16–64 letters, digits or dashes that identifies the caller's own profiles.

`POST /clone` (multipart form):

| Field | Default | Description |
|---|---|---|
| `text` | required | Up to 500 characters; English may include sound tags |
| `language` | `en` | `en` or `ta` |
| `voice_sample` | — | Audio file (or use `profile_name`) |
| `profile_name` | — | A saved profile instead of an upload |
| `style` | `spoken` | Tamil only: `spoken` (everyday) or `formal` (as written) |
| `ref_text` | — | Tamil only: transcript of the sample (detected automatically when empty) |
| `speed` | `1.0` | 0.5–2.0 |
| `consent` | — | `1` confirms you own the voice or have permission (required on Spaces) |
| `ticket` | — | Any id of 8–64 letters, digits or dashes; poll `GET /queue/{ticket}` for the place in line |

Example:

```bash
curl -X POST http://localhost:8000/clone \
  -F "text=Hello! This is my cloned voice [chuckle]." \
  -F "language=en" \
  -F "consent=1" \
  -F "voice_sample=@my_voice.wav" \
  -o cloned.wav
```

## Performance

Measured on the Hugging Face free CPU tier (2 vCPU, 16 GB RAM):

| Task | Time |
|---|---|
| English, one short sentence | ~25 s |
| Voice-to-voice, 5 s recording | ~20 s |
| Tamil, one sentence | several minutes on CPU; seconds on a GPU |

## Best results

1. Use 15–30 seconds of natural speech in a quiet room.
2. One speaker only; music and background noise are removed, but a clean recording is still best.
3. Speak at your normal pace: the clone copies the rhythm and style of the sample.
4. Write Tamil text with punctuation; full stops and commas become natural pauses.

## Troubleshooting

| Problem | Fix |
|---|---|
| Tamil is unavailable / `gated repo` error | Accept the IndicF5 terms and set `HF_TOKEN` |
| `Could not read this MP3 file` | Install ffmpeg and add it to `PATH` |
| `Unsupported file` | Upload WAV, MP3, M4A, OGG, FLAC or WebM audio |
| `Limit reached` / `busy right now` | The public site's fair-use limits; wait a few minutes |
| Server stops right after "Download complete" | Not enough free RAM; close other applications |
| `ModuleNotFoundError: backend` | Start uvicorn from the repository root |
| Words missing in Tamil | Leave the transcript empty so it is detected from the sample |
| Voice sounds noisy or muffled | Use a cleaner, longer sample (15–30 s) |

## Data and privacy

- Running locally, nothing leaves your machine except the one-time model downloads.
- On a Space, saved voices belong to the browser that saved them and are not shown to anyone else. Space storage is temporary, so they are cleared when the Space restarts.
- When `VOXAI_LOG_REPO` is set, each job's voice sample, result, text, time and an anonymous user id (a salted hash, never the IP address) are stored in that **private** Hugging Face dataset, for quality review. If you host VOXAI for others, tell your users about this in your privacy policy.

## Responsible use

Clone only your own voice, or a voice you have explicit permission to use. Do not use VOXAI to impersonate people, mislead listeners or commit fraud. English and voice-to-voice audio carries an inaudible watermark (Resemble AI Perth) that identifies it as synthetic.

## License

The VOXAI source code is released under the MIT License. The speech models it downloads (Chatterbox, IndicF5, Whisper, Demucs) are covered by their own licenses; see each model card before commercial use.

## Author

**Deepak Roshan** · AI Engineer
[deepakroshan380@gmail.com](mailto:deepakroshan380@gmail.com) · [GitHub @deepakroshan11](https://github.com/deepakroshan11)
