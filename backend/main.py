"""
VOXAI Voice Clone — Backend (Chatterbox Edition)
================================================
Model  : chatterbox-tts by Resemble AI (MIT licence, free, local, no API key)
Install: pip install chatterbox-tts
Docs   : https://github.com/resemble-ai/chatterbox

Why Chatterbox over XTTS v2?
  • Better accent / timbre preservation for non-native English speakers
  • Works on CPU (slower but fully functional)
  • Exaggeration & CFG controls for naturalness tuning
  • Simple API: model.generate(text, audio_prompt_path=...)
"""

import os
import uuid
import wave
import logging
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger("voxai")

BASE_DIR     = Path(__file__).parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
OUTPUT_DIR   = BASE_DIR / "outputs"
PROFILE_DIR  = BASE_DIR / "profiles"

OUTPUT_DIR.mkdir(exist_ok=True)
PROFILE_DIR.mkdir(exist_ok=True)

TARGET_SR      = 22050
MIN_DURATION_S = 3

# ──────────────────────────────────────────────────
app = FastAPI(title="VOXAI — Chatterbox Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")

# ──────────────────────────────────────────────────
# Chatterbox model
# ──────────────────────────────────────────────────
cb_model = None
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SR = None

@app.on_event("startup")
async def load_model():
    global cb_model, MODEL_SR
    try:
        from chatterbox.tts import ChatterboxTTS
        logger.info("Loading Chatterbox TTS on %s …", DEVICE)
        cb_model = ChatterboxTTS.from_pretrained(device=DEVICE)
        MODEL_SR = cb_model.sr
        logger.info("✅ Chatterbox ready  (sr=%d, device=%s)", MODEL_SR, DEVICE)
    except ImportError:
        logger.error("❌ chatterbox-tts not installed. Run:  pip install chatterbox-tts")
    except Exception as e:
        logger.error("❌ Failed to load Chatterbox: %s", e)


# ──────────────────────────────────────────────────
# Audio helpers
# ──────────────────────────────────────────────────

def _ffmpeg_bin():
    for c in ("ffmpeg", "ffmpeg.exe"):
        try:
            r = subprocess.run([c, "-version"],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            if r.returncode == 0:
                return c
        except FileNotFoundError:
            pass
    return None

FFMPEG = _ffmpeg_bin()


def _via_ffmpeg(src: str, dst: str) -> bool:
    if not FFMPEG:
        return False
    try:
        r = subprocess.run(
            [FFMPEG, "-y", "-i", src,
             "-ac", "1", "-ar", str(TARGET_SR), "-sample_fmt", "s16",
             "-f", "wav", dst],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120,
        )
        return r.returncode == 0 and Path(dst).exists()
    except Exception as e:
        logger.warning("ffmpeg error: %s", e)
        return False


def _via_torchaudio(src: str, dst: str) -> bool:
    try:
        waveform, sr = torchaudio.load(src)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        torchaudio.save(dst, waveform, TARGET_SR)
        return Path(dst).exists()
    except Exception as e:
        logger.warning("torchaudio error: %s", e)
        return False


def _trim_silence(waveform, sr, top_db=40.0, frame_ms=25):
    frame_len = int(sr * frame_ms / 1000)
    samples   = waveform.squeeze(0).cpu().numpy()
    n_frames  = len(samples) // frame_len
    if n_frames < 2:
        return waveform
    rms    = np.array([np.sqrt(np.mean(samples[i*frame_len:(i+1)*frame_len]**2))
                       for i in range(n_frames)])
    rms_db = 20 * np.log10(rms + 1e-9)
    thresh = rms_db.max() - top_db
    voiced = np.where(rms_db >= thresh)[0]
    if len(voiced) == 0:
        return waveform
    pad   = int(sr * 0.05)
    start = max(0, int(voiced[0] * frame_len) - pad)
    end   = min(len(samples), int((voiced[-1]+1) * frame_len) + pad)
    return torch.from_numpy(samples[start:end]).unsqueeze(0)


def enhance_audio(path: str) -> None:
    """Mono + resample + high-pass + silence-trim + normalise. In-place."""
    try:
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        try:
            waveform = torchaudio.functional.highpass_biquad(
                waveform, TARGET_SR, cutoff_freq=80.0)
        except Exception:
            pass
        waveform = _trim_silence(waveform, TARGET_SR)
        peak = waveform.abs().max()
        if peak > 1e-6:
            waveform = waveform * (10 ** (-3 / 20) / peak)
        dur = waveform.shape[1] / TARGET_SR
        if dur < MIN_DURATION_S:
            logger.warning("Reference only %.1fs — 6–20s gives best results", dur)
        else:
            logger.info("Reference audio: %.1fs ✓", dur)
        torchaudio.save(path, waveform, TARGET_SR)
    except Exception as e:
        logger.warning("Enhancement skipped: %s", e)


_NEEDS_FFMPEG = {".webm", ".mp3", ".m4a", ".aac", ".opus", ".wma", ".amr"}

def prepare_audio(file_bytes: bytes, filename: str) -> str:
    """Save → convert → enhance → return clean WAV path (caller deletes)."""
    ext = Path(filename).suffix.lower() if filename else ".bin"
    ext_map = {".opus": ".ogg", ".webm": ".webm", ".mp3": ".mp3",
               ".m4a": ".m4a", ".aac": ".aac", ".ogg": ".ogg",
               ".wav": ".wav", ".flac": ".flac"}
    ext = ext_map.get(ext, ext or ".bin")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        raw_path = tmp.name

    clean_path = raw_path + "_clean.wav"
    try:
        converted = False
        if _via_ffmpeg(raw_path, clean_path):
            converted = True
        elif ext not in _NEEDS_FFMPEG and _via_torchaudio(raw_path, clean_path):
            converted = True

        if not converted:
            raise RuntimeError(
                f"Cannot convert '{ext}' without ffmpeg.\n"
                "Install: https://www.gyan.dev/ffmpeg/builds/\n"
                "Then add ffmpeg/bin to PATH and restart the server."
            )
        enhance_audio(clean_path)
    finally:
        try:
            os.unlink(raw_path)
        except OSError:
            pass

    return clean_path


# ──────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "model":        "chatterbox-tts",
        "model_loaded": cb_model is not None,
        "device":       DEVICE,
        "ffmpeg":       FFMPEG is not None,
        "sample_rate":  MODEL_SR,
    }


# ──────────────────────────────────────────────────
# Clone
# ──────────────────────────────────────────────────
@app.post("/clone")
async def clone(
    text:         str        = Form(...),
    language:     str        = Form("en"),
    emotion:      str        = Form("neutral"),
    speed:        float      = Form(1.0),
    exaggeration: float      = Form(0.5),
    cfg_weight:   float      = Form(0.5),
    voice_sample: UploadFile = File(None),
    profile_name: str        = Form(None),
):
    if cb_model is None:
        raise HTTPException(503, "Model still loading — please wait a moment")
    if not text.strip():
        raise HTTPException(400, "Text cannot be empty")

    tmp_speaker = None
    try:
        # resolve voice reference
        if profile_name:
            sp = PROFILE_DIR / f"{profile_name}.wav"
            if not sp.exists():
                raise HTTPException(404, f"Profile '{profile_name}' not found")
            speaker_wav = str(sp)
        elif voice_sample:
            raw         = await voice_sample.read()
            fname       = voice_sample.filename or "upload.wav"
            speaker_wav = prepare_audio(raw, fname)
            tmp_speaker = speaker_wav
        else:
            raise HTTPException(400, "No voice source — upload audio or pick a profile")

        # generate
        out_path = OUTPUT_DIR / f"{uuid.uuid4().hex}.wav"

        wav_tensor = cb_model.generate(
            text,
            audio_prompt_path = speaker_wav,
            exaggeration       = float(exaggeration),
            cfg_weight         = float(cfg_weight),
        )

        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)

        sr = MODEL_SR or TARGET_SR

        # speed adjustment via resampling trick
        if abs(speed - 1.0) > 0.05:
            new_sr = int(sr * speed)
            wav_tensor = torchaudio.functional.resample(wav_tensor, new_sr, sr)

        torchaudio.save(str(out_path), wav_tensor.cpu(), sr)
        logger.info("✅ Generated %s", out_path.name)

        return FileResponse(str(out_path), media_type="audio/wav",
                            filename="cloned_voice.wav")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Clone error: %s", e)
        raise HTTPException(500, f"Synthesis failed: {e}")
    finally:
        if tmp_speaker:
            try:
                os.unlink(tmp_speaker)
            except OSError:
                pass


# ──────────────────────────────────────────────────
# Profiles
# ──────────────────────────────────────────────────
@app.post("/save-profile")
async def save_profile(
    profile_name: str        = Form(...),
    voice_sample: UploadFile = File(...),
):
    raw   = await voice_sample.read()
    fname = voice_sample.filename or "upload.wav"
    tmp   = prepare_audio(raw, fname)
    dest  = PROFILE_DIR / f"{profile_name}.wav"
    os.replace(tmp, dest)
    return {"status": "saved", "profile": profile_name}


@app.get("/profiles")
async def list_profiles():
    return {"profiles": sorted(f.stem for f in PROFILE_DIR.glob("*.wav"))}


@app.delete("/profiles/{name}")
async def delete_profile(name: str):
    p = PROFILE_DIR / f"{name}.wav"
    if p.exists():
        p.unlink()
    return {"status": "deleted", "profile": name}


# ──────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)