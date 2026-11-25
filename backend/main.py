import os, re, json, uuid, logging, tempfile, threading, subprocess
from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy import signal as sp
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq, irfft
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
import uvicorn
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling modules, any launch dir
from tamil_text import prepare_for_tts
import asr
import voiceprep as vp
import activity
from urllib.parse import quote as _quote

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger("voxai")

# ── Path resolution: works locally AND inside Docker / HF Spaces ──────────────
# When running as "uvicorn backend.main:app" from /app, __file__ is /app/backend/main.py
BASE_DIR     = Path(__file__).parent.parent          # /app  (or voice-clone/ locally)
FRONTEND_DIR = BASE_DIR / "frontend"
OUTPUT_DIR   = BASE_DIR / "outputs"
PROFILE_DIR  = BASE_DIR / "profiles"
OUTPUT_DIR.mkdir(exist_ok=True)
PROFILE_DIR.mkdir(exist_ok=True)
REF_CACHE_DIR = OUTPUT_DIR / ".voice-prompts"   # prepared Tamil prompts (see _indic_reference)

CHATTERBOX_SR    = 24000
OUTPUT_SR        = 48000   # v8: output at 48kHz (broadcast quality)
MIN_DURATION_S   = 4
BEST_DURATION_S  = 15
WARN_DURATION_S  = 10
CHUNK_CHAR_LIMIT = 250     # whole sentences together keep intonation flowing (Turbo: ~15 s)
# Output processing: "natural" = clean resample + loudness + smooth limiter (keeps the
# speaker's own pitch and texture, like YouTube/Meta dubbing); "studio" = the old
# 7-stage chain (its pitch-band "de-noising" and comb reverb made voices robotic).
POST_MODE = os.environ.get("VOXAI_POST", "natural").strip().lower()

# v8: UPDATED DEFAULTS -- lower exaggeration = stable pitch, less creak
DEFAULT_EXAGGERATION = 0.3
DEFAULT_CFG_WEIGHT   = 0.7

# ── Speed settings ────────────────────────────────────────────────────────────
# English engine: "turbo" = Chatterbox-Turbo (no CFG pass, 2-step decoder; much
# faster on CPU, ignores the exaggeration/cfg sliders, needs a >5s sample) or
# "standard" = original Chatterbox.  Set VOXAI_EN_ENGINE=standard to switch back.
EN_ENGINE = os.environ.get("VOXAI_EN_ENGINE", "turbo").strip().lower()
TURBO_REPO = "ResembleAI/chatterbox-turbo"
TURBO_FILES = ["t3_turbo_v1.safetensors", "s3gen_meanflow.safetensors", "ve.safetensors",
               "*.json", "*.txt", "conds.pt"]   # skips the unused 1GB s3gen.safetensors
TURBO_MIN_REF_S = 5.0
# IndicF5 flow-matching steps: 32 = model default (best), 16 = ~2x faster.
INDIC_NFE_STEPS = int(os.environ.get("VOXAI_INDIC_STEPS", "32"))
# CPU threads for PyTorch (0 = PyTorch default).
TORCH_THREADS = int(os.environ.get("VOXAI_THREADS", "0"))
if TORCH_THREADS > 0:
    torch.set_num_threads(TORCH_THREADS)

# ── Indic languages (IndicF5, AI4Bharat) ──────────────────────────────────────
# IndicF5 needs the reference clip AND its exact transcript, so the reference is
# never cropped (cropping would desync it from the transcript).
INDIC_REPO      = os.environ.get("INDIC_REPO", "ai4bharat/IndicF5")
INDIC_SR        = 24000
# Only Tamil is enabled for now. IndicF5 also supports: hi Hindi, te Telugu,
# kn Kannada, ml Malayalam, bn Bengali, mr Marathi, gu Gujarati, pa Punjabi,
# or Odia, as Assamese -- add them back here (and in the frontend) to enable.
INDIC_LANGS     = {"ta": "Tamil"}
INDIC_MIN_REF_S = 3
INDIC_MAX_REF_S = 60      # longer samples are fine: they are clipped at a pause and re-transcribed
INDIC_REF_MAX_S = 12.0    # F5-TTS README: prompts under ~12 s
INDIC_REF_PAD_S = 0.1     # short tail only: with 0.5 s IndicF5 continued the silence for ~1 s,
                          # then rushed the sentence and dropped its first word (tested)
INDIC_DEFAULT_RATE = 15.0 # Tamil letters+signs per second, measured on AI4Bharat's Tamil prompt;
                          # with it an English sample gave the same ASR accuracy as a Tamil one
INDIC_CHUNK_CHARS = 120
MAX_TEXT_CHARS  = 500
# Reading pace relative to the speaker's own speaking rate.  Measured: at the sample's
# rate (a fast, casual film clip) the Tamil clone spoke ~6.9 syllables/s -- "fast"
# speech in reading studies (normal read speech: ~4.3-5.0 syll/s including pauses).
INDIC_PACE      = float(os.environ.get("VOXAI_PACE", "0.9"))
INDIC_RATE_MIN, INDIC_RATE_MAX = 11.0, 17.5   # Tamil letters+signs per second of speech
PERFORM_MAX_S   = 60      # voice-to-voice: longest performance accepted
PROMPT_TAG      = "|prompt-v2"   # transcript cache tag (v2: made from the isolated voice)

# One synthesis at a time: models are not thread-safe and the free CPU is tiny.
SYNTH_LOCK = threading.Lock()

PROFILE_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")

def _safe_profile(name: str) -> str:
    """Reject names that could escape PROFILE_DIR (e.g. '../../x')."""
    name = (name or "").strip()
    if not PROFILE_RE.match(name):
        raise HTTPException(400, "Profile name may only use letters, digits, '-' and '_' (max 40)")
    return name

def _profile_meta(name: str) -> dict:
    p = PROFILE_DIR / f"{name}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}

app = FastAPI(title="VOXAI v8 -- Definitive Fix")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["X-Quality-Warning", "X-Spoken-Text", "X-Ref-Text", "X-Ref-Replaced",
                                   "X-Voice-Prep", "X-Rhythm"])
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
async def root():
    # Safe fallback: works locally AND on HF Spaces / Docker where frontend may be absent
    idx = FRONTEND_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return {"status": "VOXAI backend running", "docs": "/docs"}

cb_model = None
cb_engine = None          # "turbo" or "standard" -- whichever actually loaded
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SR = None

def _load_turbo():
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    from huggingface_hub import snapshot_download
    path = snapshot_download(TURBO_REPO, allow_patterns=TURBO_FILES)
    return ChatterboxTurboTTS.from_local(path, DEVICE)

def _load_standard():
    from chatterbox.tts import ChatterboxTTS
    return ChatterboxTTS.from_pretrained(device=DEVICE)

@app.on_event("startup")
async def load_model():
    global cb_model, cb_engine, MODEL_SR
    logger.info("CPU threads: %d", torch.get_num_threads())
    order = ["turbo", "standard"] if EN_ENGINE == "turbo" else ["standard"]
    for engine in order:
        try:
            logger.info("Loading Chatterbox (%s) on %s ...", engine, DEVICE)
            cb_model = _load_turbo() if engine == "turbo" else _load_standard()
            cb_engine = engine
            MODEL_SR = cb_model.sr
            logger.info("Chatterbox (%s) ready  sr=%d  device=%s", engine, MODEL_SR, DEVICE)
            break
        except Exception as e:
            logger.error("Chatterbox (%s) failed: %s", engine, e)
    # Warm IndicF5 in the background so the first Tamil request isn't a 1.4GB download.
    if os.environ.get("PRELOAD_INDIC", "1") == "1":
        threading.Thread(target=_safe_load_indic, daemon=True).start()


# ======================================================================
#  INDICF5 (Tamil / Hindi / Telugu / ...) -- lazy, thread-safe loader
# ======================================================================

indic_model = None
indic_error = None
_indic_load_lock = threading.Lock()

def get_indic_model():
    """Load IndicF5 once. The HF repo is gated: accept its terms on huggingface.co
    and set HF_TOKEN (env var / Space secret), otherwise loading fails with 401."""
    global indic_model, indic_error
    if indic_model is not None:
        return indic_model
    with _indic_load_lock:
        if indic_model is None:
            from transformers import AutoConfig
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            logger.info("Loading IndicF5 (%s) ...", INDIC_REPO)
            # AutoModel.from_pretrained() in transformers 5.x builds the model on the
            # "meta" device, but INF5Model.__init__ loads the vocoder/DiT eagerly and
            # crashes ("Tensor.item() cannot be called on meta tensors"). Build it for
            # real, then load the checkpoint ourselves.
            cfg = AutoConfig.from_pretrained(INDIC_REPO, trust_remote_code=True)
            cls = get_class_from_dynamic_module("model.INF5Model", INDIC_REPO)
            m = cls(cfg)
            state = load_file(hf_hub_download(INDIC_REPO, "model.safetensors"))
            missing, unexpected = m.load_state_dict(state, strict=False)
            if missing or unexpected:
                logger.warning("IndicF5 weights: %d missing, %d unexpected keys",
                               len(missing), len(unexpected))
            del state
            # INF5Model wraps its sub-models in torch.compile; on CPU (esp. Windows,
            # which needs a C++ toolchain for inductor) run the plain modules instead.
            if DEVICE == "cpu":
                m.ema_model = getattr(m.ema_model, "_orig_mod", m.ema_model)
                m.vocoder = getattr(m.vocoder, "_orig_mod", m.vocoder)
            m.eval()
            try:
                m = m.to(DEVICE)
            except Exception:
                pass
            indic_model = m
            indic_error = None
            logger.info("IndicF5 ready")
    return indic_model

def _safe_load_indic():
    global indic_error
    try:
        get_indic_model()
    except Exception as e:
        indic_error = str(e)
        logger.error("IndicF5 preload failed: %s", e)

def _norm_ref_text(t: str) -> str:
    """Transcript of the voice prompt, ending in a run of pause marks.
    Samples often end mid-word, and the recogniser misses that last fragment. F5 then
    'reads' the first word of the NEW text into the sample's time slot, and it is cut
    away with the sample (the swallowed first word). The pause marks absorb such a
    fragment instead; whatever F5 aligns inside the sample slot is discarded anyway,
    so with an exact transcript they only add a short pause, which is trimmed."""
    t = re.sub(r"\s+", " ", (t or "")).strip().rstrip(".")
    return t + ". . . . . "

INDIC_PAUSE_MARK_S = 0.4   # extra time given for that pause so the sentence isn't rushed


# ======================================================================
#  FFMPEG
# ======================================================================

def _ffmpeg_bin():
    for c in ("ffmpeg", "ffmpeg.exe"):
        try:
            if subprocess.run([c, "-version"], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0:
                return c
        except FileNotFoundError:
            pass
    return None

FFMPEG = _ffmpeg_bin()
_NEEDS_FFMPEG = {".webm", ".mp3", ".m4a", ".aac", ".opus", ".wma", ".amr"}

def _ffmpeg_convert(src: str, dst: str) -> bool:
    if not FFMPEG:
        return False
    try:
        r = subprocess.run(
            [FFMPEG, "-y", "-i", src, "-ac", "1", "-sample_fmt", "s16", "-f", "wav", dst],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120
        )
        return r.returncode == 0 and Path(dst).exists()
    except Exception as e:
        logger.warning("ffmpeg: %s", e)
        return False

def _torchaudio_convert(src: str, dst: str) -> bool:
    try:
        w, sr = torchaudio.load(src)
        if w.shape[0] > 1: w = w.mean(0, keepdim=True)
        torchaudio.save(dst, w, sr)
        return Path(dst).exists()
    except Exception as e:
        logger.warning("torchaudio convert: %s", e)
        return False


# ======================================================================
#  REFERENCE AUDIO PREPARATION
# ======================================================================

def _best_segment(wav: torch.Tensor, sr: int, target_s: float) -> torch.Tensor:
    s = wav.squeeze(0)
    if s.shape[0] / sr <= target_s + 1:
        return wav
    tgt  = int(target_s * sr)
    step = int(0.25 * sr)
    best_rms, best_start = -1.0, 0
    for start in range(0, s.shape[0] - tgt, step):
        rms = float(torch.sqrt(torch.mean(s[start:start + tgt] ** 2)))
        if rms > best_rms:
            best_rms, best_start = rms, start
    return s[best_start:best_start + tgt].unsqueeze(0)

def _gentle_trim(wav: torch.Tensor, sr: int, top_db: float = 28.0) -> torch.Tensor:
    fl = int(sr * 0.02)
    s  = wav.squeeze(0).cpu().numpy()
    n  = len(s) // fl
    if n < 4: return wav
    rms = np.array([np.sqrt(np.mean(s[i*fl:(i+1)*fl]**2)) for i in range(n)])
    db  = 20 * np.log10(rms + 1e-9)
    v   = np.where(db >= db.max() - top_db)[0]
    if not len(v): return wav
    pad = int(sr * 0.1)
    return torch.from_numpy(
        s[max(0, v[0]*fl - pad): min(len(s), (v[-1]+1)*fl + pad)]
    ).unsqueeze(0)

def analyse_reference(path: str) -> dict:
    """Measure RMS and HF/LF ratio from reference for loudness matching."""
    try:
        w, sr = torchaudio.load(path)
        if w.shape[0] > 1: w = w.mean(0, keepdim=True)
        s = w.squeeze(0).cpu().numpy().astype(np.float64)
        rms    = float(np.sqrt(np.mean(s**2)))
        sos_hf = sp.butter(4, min(4000.0/(sr/2), 0.99), 'high', output='sos')
        sos_lf = sp.butter(4, min(300.0/(sr/2),  0.99), 'low',  output='sos')
        hf_rms = float(np.sqrt(np.mean(sp.sosfilt(sos_hf, s)**2)))
        lf_rms = float(np.sqrt(np.mean(sp.sosfilt(sos_lf, s)**2)))
        return {"rms": rms, "hf_lf_ratio": hf_rms / (lf_rms + 1e-9)}
    except Exception as e:
        logger.warning("Reference analysis: %s", e)
        return {"rms": 0.035, "hf_lf_ratio": 0.11}

def enhance_reference(path: str, crop: bool = True) -> dict:
    try:
        w, sr = torchaudio.load(path)
        if w.shape[0] > 1: w = w.mean(0, keepdim=True)
        if crop:   # cleanest <=15 s stretch, cut inside pauses (not mid-word)
            s0, _, _ = vp.pick_window(w.squeeze(0).numpy(), sr, BEST_DURATION_S, min_s=8.0)
            w = torch.from_numpy(np.ascontiguousarray(s0, dtype=np.float32)).unsqueeze(0)
        w   = _gentle_trim(w, sr, top_db=28.0)
        s   = w.squeeze(0).cpu().numpy().astype(np.float64)
        sos = sp.butter(2, min(60.0/(sr/2), 0.99), btype='highpass', output='sos')
        s   = sp.sosfilt(sos, s)
        w   = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
        pk  = w.abs().max()
        if pk > 1e-6: w = w * (10**(-8/20) / pk)
        dur = w.shape[1] / sr
        torchaudio.save(path, w, sr)
        info = analyse_reference(path)
        info["duration"] = dur
        info["ok"]       = dur >= MIN_DURATION_S
        return info
    except Exception as e:
        logger.warning("Reference enhance: %s", e)
        return {"duration": 0, "ok": False, "rms": 0.035, "hf_lf_ratio": 0.11}

def prepare_audio(file_bytes: bytes, filename: str, crop: bool = True) -> tuple:
    ext = Path(filename).suffix.lower() if filename else ".bin"
    ext_map = {".opus": ".ogg", ".webm": ".webm", ".mp3": ".mp3", ".m4a": ".m4a",
               ".aac": ".aac", ".ogg": ".ogg", ".wav": ".wav", ".flac": ".flac"}
    ext = ext_map.get(ext, ext or ".bin")
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes); raw = tmp.name
    clean = raw + "_ref.wav"
    try:
        ok = _ffmpeg_convert(raw, clean) or \
             (ext not in _NEEDS_FFMPEG and _torchaudio_convert(raw, clean))
        if not ok:
            raise RuntimeError(f"Cannot convert '{ext}'. Install ffmpeg.")
        info = enhance_reference(clean, crop=crop)
        if info["duration"] < MIN_DURATION_S:
            raise RuntimeError(
                f"Too short ({info['duration']:.1f}s, min {MIN_DURATION_S}s). "
                "Record 15-30 seconds for best quality.")
    finally:
        try: os.unlink(raw)
        except OSError: pass
    return clean, info


# ======================================================================
#  STAGE 1: BASS RESTORATION
#  Measured: 100-300Hz is 9-11dB too quiet in ALL cloned outputs.
# ======================================================================

def apply_bass_restoration(s: np.ndarray, sr: int) -> np.ndarray:
    """
    Measured deficit: 100Hz=-11dB, 150Hz=-9.3dB, 200Hz=-11dB, 250Hz=-8.4dB.
    Low shelf +10dB below 300Hz. Body bump +3dB at 450-600Hz.
    Applied additively (parallel blend) to avoid phase issues.
    """
    s = s.astype(np.float64)

    # Low shelf +10dB below 300Hz
    sos_ls = sp.butter(4, min(300.0 / (sr / 2), 0.99), btype='low', output='sos')
    bass   = sp.sosfilt(sos_ls, s)
    s      = s + (10 ** (10.0 / 20) - 1.0) * bass

    # Body bump +3dB at 450-600Hz
    lo    = max(0.001, 420.0/(sr/2))
    hi    = min(0.999, 650.0/(sr/2))
    sos_b = sp.butter(2, [lo, hi], btype='bandpass', output='sos')
    body  = sp.sosfilt(sos_b, s)
    s     = s + (10 ** (3.0/20) - 1.0) * body

    return s.astype(np.float32)


# ======================================================================
#  STAGE 2: CEPSTRAL LIFTERING
#  Remove vocoder inter-harmonic noise (ZCR 2.44x too high).
#  Standard technique in CELP codecs and speech enhancement.
# ======================================================================

def apply_cepstral_denoising(s: np.ndarray, sr: int) -> np.ndarray:
    """
    Cepstral liftering: zero out quefrency bins that correspond to
    inter-harmonic noise while preserving harmonic peaks and vocal tract envelope.

    Process per overlapping frame:
    1. Real cepstrum = IFFT(log|STFT|)
    2. Apply lifter window (suppress noise quefrency zone)
    3. Reconstruct via FFT -> exp -> recombine with original phase
    4. Overlap-add synthesis
    """
    s          = s.astype(np.float64)
    frame_size = 512
    hop        = 256
    result     = np.zeros(len(s) + frame_size)
    norm       = np.zeros(len(s) + frame_size)
    window     = np.hanning(frame_size)

    lifter_lo = 10
    lifter_hi = int(sr / 80.0)   # ~300 samples at 24kHz

    for i in range(0, len(s) - frame_size, hop):
        frame = s[i:i + frame_size] * window

        if np.sqrt(np.mean(frame**2)) < 0.002:
            result[i:i + frame_size] += frame
            norm[i:i + frame_size]   += window
            continue

        spec     = rfft(frame, n=frame_size)
        log_spec = np.log(np.abs(spec) + 1e-9)
        cepstrum = np.real(np.fft.ifft(
            np.concatenate([log_spec, log_spec[-2:0:-1]])
        ))[:frame_size]

        lifter = np.ones(frame_size)
        lifter[lifter_lo:lifter_hi] = np.linspace(1.0, 0.2, lifter_hi - lifter_lo)
        lifter[frame_size//2:] = lifter[frame_size//2:0:-1]

        cepstrum_f = cepstrum * lifter
        log_spec_s = np.real(np.fft.fft(cepstrum_f))[:frame_size//2 + 1]
        spec_out   = np.exp(log_spec_s) * np.exp(1j * np.angle(spec))
        frame_out  = np.real(irfft(spec_out, n=frame_size)) * window

        result[i:i + frame_size] += frame_out
        norm[i:i + frame_size]   += window ** 2

    norm   = np.maximum(norm, 1e-6)
    result = (result[:len(s)] / norm[:len(s)])

    # Blend 70% processed + 30% original
    return (result * 0.70 + s * 0.30).astype(np.float32)


# ======================================================================
#  STAGE 3: HARMONIC BANDWIDTH EXTENSION + UPSAMPLE
#  Model outputs max 12kHz. Reference has content to 18kHz.
#  Uses Spectral Band Replication (same principle as MP3 SBR, AAC HE).
# ======================================================================

def apply_bandwidth_extension(s: np.ndarray, src_sr: int, dst_sr: int = 48000) -> tuple:
    """
    Harmonic bandwidth extension (HBE):
    1. Upsample to 48kHz via torchaudio (correct anti-imaging)
    2. Extract 4-11kHz seed band
    3. Square (generates 2x frequency -> shifts 4-11kHz to 8-22kHz)
    4. Band-pass to 12-20kHz only
    5. Roll off with natural -12dB/oct speech tilt
    6. Mix at -18dB (barely audible -- just removes the hard cutoff)
    """
    w_t  = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)
    w_up = torchaudio.functional.resample(w_t, src_sr, dst_sr)
    s_up = w_up.squeeze(0).numpy().astype(np.float64)

    # Seed band: 4-11kHz
    sos_seed = sp.butter(6,
        [max(0.001, 4000.0/(dst_sr/2)), min(0.999, 11000.0/(dst_sr/2))],
        btype='bandpass', output='sos')
    seed = sp.sosfilt(sos_seed, s_up)

    # Harmonic generation via squaring
    hf = seed ** 2
    pk = np.abs(hf).max()
    if pk > 1e-9: hf /= pk

    # Band-pass 12-20kHz
    sos_hf = sp.butter(6,
        [max(0.001, 12000.0/(dst_sr/2)), min(0.999, 20000.0/(dst_sr/2))],
        btype='bandpass', output='sos')
    hf = sp.sosfilt(sos_hf, hf)

    # Natural speech tilt
    sos_tilt = sp.butter(2, min(16000.0/(dst_sr/2), 0.99), btype='low', output='sos')
    hf = sp.sosfilt(sos_tilt, hf)

    # Mix at -18dB
    s_out = s_up + hf * (10 ** (-18.0 / 20))
    return s_out.astype(np.float32), dst_sr


# ======================================================================
#  STAGE 4: NATURAL MULTIBAND DYNAMICS
# ======================================================================

def apply_natural_dynamics(s: np.ndarray, sr: int) -> np.ndarray:
    """
    Gentle 3-band compression (2:1) to balance the voice naturally.
    Not heavy-handed -- just controls peaks and lifts presence.
    """
    try:
        s = s.astype(np.float64)

        def compress_band(band, thr_db=-20.0, ratio=2.0, att_ms=8.0, rel_ms=100.0, mk_db=1.5):
            thr = 10 ** (thr_db / 20)
            att = np.exp(-1.0 / (sr * att_ms / 1000))
            rel = np.exp(-1.0 / (sr * rel_ms / 1000))
            mk  = 10 ** (mk_db / 20)
            out = np.empty_like(band)
            env = 0.0; g = 1.0
            for i, x in enumerate(band):
                lv  = abs(x)
                env = lv + (att if lv > env else rel) * (env - lv)
                g_t = ((thr * (env/thr)**(1/ratio)) / (env + 1e-10)) if env > thr else 1.0
                g  += 0.008 * (g_t - g)
                out[i] = x * g * mk
            return out

        lo_sos  = sp.butter(4, [max(0.001, 80/(sr/2)),   min(0.999, 400/(sr/2))],  'bandpass', output='sos')
        mid_sos = sp.butter(4, [max(0.001, 400/(sr/2)),  min(0.999, 3000/(sr/2))], 'bandpass', output='sos')
        hi_sos  = sp.butter(4, [max(0.001, 3000/(sr/2)), min(0.999, 8000/(sr/2))], 'bandpass', output='sos')

        lo  = sp.sosfilt(lo_sos,  s)
        mid = sp.sosfilt(mid_sos, s)
        hi  = sp.sosfilt(hi_sos,  s)
        res = s - lo - mid - hi

        return (res
                + compress_band(lo,  thr_db=-22, ratio=2.0, mk_db=2.0)
                + compress_band(mid, thr_db=-20, ratio=2.0, mk_db=1.5)
                + compress_band(hi,  thr_db=-18, ratio=2.5, mk_db=2.5)
               ).astype(np.float32)
    except Exception as e:
        logger.warning("Multiband compress: %s", e)
        return s.astype(np.float32)


# ======================================================================
#  STAGE 5: PHONE ROOM TONE
#  Matched to WhatsApp/phone recording (RT60 ~60ms)
# ======================================================================

def apply_room_tone(s: np.ndarray, sr: int, wet: float = 0.05) -> np.ndarray:
    """
    5% wet subtle room -- removes the dead-dry quality without sounding
    like reverb. Matched to hand-held phone recording environment.
    """
    try:
        s   = s.astype(np.float64)
        pre = int(sr * 0.006)
        delayed = np.zeros_like(s)
        if pre < len(s): delayed[pre:] = s[:-pre]

        combs = []
        for dm, g_val in [(27.1, 0.76), (32.3, 0.76), (38.7, 0.74)]:
            d   = max(1, int(sr * dm / 1000))
            g, buf, bi = g_val, np.zeros(d), 0
            out = np.empty_like(delayed)
            for n, x in enumerate(delayed):
                out[n]  = buf[bi]
                buf[bi] = x + g * buf[bi]
                bi      = (bi + 1) % d
            combs.append(out)

        reverb = sum(combs) / len(combs)
        for dm, g in [(4.5, 0.62), (1.4, 0.62)]:
            d       = max(1, int(sr * dm / 1000))
            buf, bi = np.zeros(d), 0
            out     = np.empty_like(reverb)
            for n, x in enumerate(reverb):
                ds      = buf[bi]; inp = x + g * ds
                out[n]  = -g * inp + ds; buf[bi] = inp; bi = (bi+1) % d
            reverb = out

        sos    = sp.butter(2, min(3500.0/(sr/2), 0.99), 'low', output='sos')
        reverb = sp.sosfilt(sos, reverb)
        return (s * (1 - wet) + reverb * wet).astype(np.float32)
    except Exception as e:
        logger.warning("Room tone: %s", e)
        return s.astype(np.float32)


# ======================================================================
#  STAGE 6: EBU R128 LOUDNESS NORMALIZATION
#  Broadcast standard: -16 LUFS / -1dBTP
# ======================================================================

def apply_lufs_normalization(s: np.ndarray, sr: int, target_lufs: float = -16.0) -> np.ndarray:
    """
    Simplified EBU R128 loudness normalization with K-weighting approximation.
    -16 LUFS matches WhatsApp voice messages, Spotify podcasts, YouTube speech.
    """
    s = s.astype(np.float64)
    # K-weighting high-shelf approximation
    sos_kw = sp.butter(2, min(1500.0/(sr/2), 0.99), 'high', output='sos')
    s_kw   = s + (10**(4.0/20) - 1.0) * sp.sosfilt(sos_kw, s)

    frame      = int(sr * 0.1)
    rms_blocks = []
    for i in range(0, len(s_kw) - frame, frame):
        rms = np.sqrt(np.mean(s_kw[i:i+frame]**2))
        if rms > 1e-4: rms_blocks.append(rms)

    if not rms_blocks:
        return s.astype(np.float32)

    rms_arr      = np.array(rms_blocks)
    mean_rms     = np.mean(rms_arr)
    gate_rms     = np.mean(rms_arr[rms_arr > mean_rms * 10**(-10/20)])
    current_lufs = 20 * np.log10(gate_rms + 1e-9) - 0.69
    gain_db      = target_lufs - current_lufs
    gain         = 10 ** (gain_db / 20)
    logger.info("LUFS: current=%.1f target=%.1f gain=%+.1fdB", current_lufs, target_lufs, gain_db)
    return np.clip(s * gain, -0.97, 0.97).astype(np.float32)


def apply_true_peak_limiter(s: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    """Smooth true-peak limiter. Prevents inter-sample peaks."""
    s       = s.astype(np.float64)
    ceiling = 10 ** (ceiling_db / 20)
    sr_approx = max(1, len(s) // 10)
    att = np.exp(-1.0 / sr_approx)
    rel = np.exp(-1.0 / (sr_approx * 10))
    out = np.empty_like(s)
    g   = 1.0
    for i, x in enumerate(s):
        lv  = abs(x)
        g_t = min(1.0, ceiling / (lv + 1e-9))
        g  += (att if g_t < g else rel) * (g_t - g)
        out[i] = x * g
    return np.clip(out, -ceiling, ceiling).astype(np.float32)


# ======================================================================
#  NATURAL CHAIN (default)
#  YouTube "expressive speech" / Meta AI translations aim to keep the speaker's
#  own pitch, intonation and pacing.  So: no spectral surgery, no synthetic
#  reverb or treble -- only what a human engineer would do to a voice note.
# ======================================================================

def _smooth_limit(s: np.ndarray, sr: int, ceiling_db: float = -1.0) -> np.ndarray:
    """Look-ahead peak limiter: gain is lowered smoothly over ~10 ms around each peak
    (the old chain hard-clipped after up to +16 dB of gain = audible distortion)."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    c = 10 ** (ceiling_db / 20)
    need = np.minimum(1.0, c / (np.abs(s) + 1e-12))
    if need.min() >= 1.0:
        return s
    w = max(1, int(sr * 0.01))
    g = uniform_filter1d(minimum_filter1d(need, size=2 * w + 1), size=w + 1)
    return np.clip(s * g, -c, c)


def natural_chain(s: np.ndarray, src_sr: int) -> tuple:
    logger.info("Post: natural (high-pass, 48 kHz, -16 LUFS, smooth limiter)")
    s = sp.sosfilt(sp.butter(2, 60.0 / (src_sr / 2), 'highpass', output='sos'), s.astype(np.float64))
    s = torchaudio.functional.resample(torch.from_numpy(s.astype(np.float32)).unsqueeze(0),
                                       src_sr, OUTPUT_SR).squeeze(0).numpy().astype(np.float64)
    try:
        import pyloudnorm as pyln
        loud = pyln.Meter(OUTPUT_SR).integrated_loudness(s)
    except Exception as e:
        logger.warning("Loudness meter: %s", e)
        loud = float("nan")
    if np.isfinite(loud):
        gain_db = float(np.clip(-16.0 - loud, -20.0, 20.0))
        logger.info("Loudness %.1f LUFS -> -16 LUFS (%+.1f dB)", loud, gain_db)
        s = s * 10 ** (gain_db / 20)
    return _smooth_limit(s, OUTPUT_SR).astype(np.float32), OUTPUT_SR


# ======================================================================
#  MASTER CHAIN ("studio", VOXAI_POST=studio)
# ======================================================================

def professional_chain(wav_path: str, src_sr: int, ref_info: dict, engine: str = "chatterbox") -> tuple:
    """
    v8 definitive chain:
    1. Bass restoration (+10dB shelf @ 100-300Hz)
    2. Cepstral de-noising (vocoder buzz removal)
    3. Harmonic bandwidth extension + 24->48kHz upsample
    4. Natural multiband dynamics (2:1, 3 bands)
    5. Room tone (5% wet, matched to phone recording)
    6. EBU R128 loudness normalization (-16 LUFS)
    7. True-peak limiter (-1dBTP)
    """
    _, raw = wavfile.read(wav_path)
    if raw.ndim > 1: raw = raw.mean(axis=1)
    s = (raw.astype(np.float32) / 32768.0) if raw.dtype == np.int16 else raw.astype(np.float32)

    if POST_MODE != "studio":
        return natural_chain(s, src_sr)

    # Stages 1-2 were tuned against measured Chatterbox artefacts (bass deficit,
    # vocoder buzz). IndicF5 uses a different vocoder (Vocos), so skip them there.
    if engine == "chatterbox":
        logger.info("Stage 1: Bass restoration")
        s = apply_bass_restoration(s, src_sr)

        logger.info("Stage 2: Cepstral de-noising")
        s = apply_cepstral_denoising(s, src_sr)

    logger.info("Stage 3: Bandwidth extension + 24->48kHz")
    s, out_sr = apply_bandwidth_extension(s, src_sr, OUTPUT_SR)

    logger.info("Stage 4: Multiband dynamics")
    s = apply_natural_dynamics(s, out_sr)

    logger.info("Stage 5: Room tone")
    s = apply_room_tone(s, out_sr, wet=0.05)

    logger.info("Stage 6: LUFS normalization")
    s = apply_lufs_normalization(s, out_sr, target_lufs=-16.0)

    logger.info("Stage 7: True-peak limiter")
    s = apply_true_peak_limiter(s, ceiling_db=-1.0)

    return s, out_sr


# ======================================================================
#  CHUNKED SYNTHESIS
# ======================================================================

def _split_text(text: str, max_chars: int = CHUNK_CHAR_LIMIT) -> list:
    """Whole sentences per chunk (keeps intonation natural); a sentence longer than
    max_chars is split at commas rather than left oversized."""
    pieces = []
    for sent in re.split(r'(?<=[.!?।])\s+', text.strip()):
        if len(sent) <= max_chars:
            pieces.append(sent)
            continue
        cur = ""
        for part in re.split(r'(?<=[,;:])\s+', sent):
            if cur and len(cur) + len(part) + 1 > max_chars:
                pieces.append(cur); cur = part
            else:
                cur = (cur + " " + part).strip()
        if cur: pieces.append(cur)
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) + 1 > max_chars:
            chunks.append(cur); cur = p
        else:
            cur = (cur + " " + p).strip()
    if cur: chunks.append(cur)
    return [c for c in chunks if c.strip()]


def _speech_mask(s: np.ndarray, sr: int, frame_s: float = 0.02, floor_db: float = -38.0):
    """Per-frame speech/silence, relative to the loudest frame (works for quiet
    phone recordings and noisy clips alike, unlike a fixed -50 dBFS)."""
    hop = max(1, int(sr * frame_s))
    n = len(s) // hop
    if n == 0:
        return np.zeros(0, bool), hop
    db = 20 * np.log10(np.sqrt((s[:n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1)) + 1e-9)
    return db > db.max() + floor_db, hop


def _tidy_pauses(s: np.ndarray, sr: int, max_pause: float = 0.35,
                 head: float = 0.08, tail: float = 0.2) -> np.ndarray:
    """Trim edge silence and shorten pauses longer than max_pause.
    Speech itself is never touched (the old split_on_silence could drop quiet words)."""
    s = np.asarray(s, np.float32).reshape(-1)
    m, hop = _speech_mask(s, sr)
    if not m.any():
        return s
    edges = np.flatnonzero(np.diff(m.astype(np.int8))) + 1
    bounds = np.concatenate([[0], edges, [len(m)]])
    runs = [(bounds[i] * hop, bounds[i + 1] * hop) for i in range(len(bounds) - 1) if m[bounds[i]]]
    keep = int(max_pause * sr) // 2
    out = [s[max(0, runs[0][0] - int(head * sr)):runs[0][0]]]
    for i, (a, b) in enumerate(runs):
        out.append(s[a:b])
        if i + 1 < len(runs):
            gap = s[b:runs[i + 1][0]]
            out.append(gap if len(gap) <= 2 * keep else np.concatenate([gap[:keep], gap[-keep:]]))
        else:
            out.append(s[b:b + int(tail * sr)])
    return np.concatenate(out)


def _join_natural(segs: list, texts: list, sr: int, rhythm: dict = None) -> np.ndarray:
    """Join chunks with the SPEAKER'S OWN pause lengths, measured from the voice sample
    (voiceprep.speaker_rhythm): a short pause after a comma, a long one after a sentence."""
    r = rhythm or vp.DEFAULT_RHYTHM
    out = []
    for i, seg in enumerate(segs):
        seg = _tidy_pauses(seg, sr, max_pause=max(0.6, 1.5 * r["sentence"]), head=0.06, tail=0.12)
        if i:
            pause = r["comma"] if texts[i - 1].rstrip()[-1:] in ",;:" else r["sentence"]
            out.append(np.zeros(int(sr * pause), np.float32))
        f = min(int(sr * 0.005), len(seg) // 2)
        if f:
            seg = seg.copy()
            seg[:f] *= np.linspace(0, 1, f, dtype=np.float32)
            seg[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
        out.append(seg)
    return np.concatenate(out)

_conds_key = None   # fingerprint of the reference currently loaded into cb_model.conds

def _prepare_voice(speaker_wav: str, exaggeration: float, voice_key: str = None):
    """Embed the reference voice once and reuse it across chunks AND requests.
    (Chatterbox's generate(audio_prompt_path=...) re-embeds it on every call.)
    voice_key identifies the *source* voice (profile name or hash of the uploaded
    bytes): the cleaned temp file differs byte-wise on every upload."""
    global _conds_key
    key = voice_key
    if not key:
        import hashlib
        with open(speaker_wav, "rb") as f:
            key = hashlib.sha1(f.read()).hexdigest()
    if cb_engine == "standard":
        key += f"|ex={exaggeration:.3f}"
    if key == _conds_key and cb_model.conds is not None:
        logger.info("Reusing cached voice embedding")
        return
    if cb_engine == "turbo":
        import librosa
        dur = librosa.get_duration(path=speaker_wav)
        if dur <= TURBO_MIN_REF_S:
            raise HTTPException(400, f"English (Turbo) needs a voice sample longer than "
                                     f"{TURBO_MIN_REF_S:.0f}s (yours is {dur:.1f}s)")
        cb_model.prepare_conditionals(speaker_wav)
    else:
        cb_model.prepare_conditionals(speaker_wav, exaggeration=exaggeration)
    _conds_key = key

def _generate_chunk(text: str, exaggeration: float, cfg_weight: float) -> torch.Tensor:
    with torch.inference_mode():
        if cb_engine == "turbo":
            w = cb_model.generate(text)            # Turbo ignores exaggeration / cfg
        else:
            w = cb_model.generate(text, exaggeration=exaggeration, cfg_weight=cfg_weight)
    return w.unsqueeze(0) if w.dim() == 1 else w

def synthesise(text: str, speaker_wav: str, exaggeration: float,
               cfg_weight: float, sr: int, voice_key: str = None,
               rhythm: dict = None) -> torch.Tensor:
    chunks = _split_text(text)
    logger.info("Synthesising %d chunks  engine=%s  ex=%.2f cfg=%.2f",
                len(chunks), cb_engine, exaggeration, cfg_weight)
    _prepare_voice(speaker_wav, exaggeration, voice_key)
    segs = []
    for i, chunk in enumerate(chunks):
        logger.info("  Chunk %d/%d: '%s...'", i+1, len(chunks), chunk[:60])
        w = _generate_chunk(chunk, exaggeration, cfg_weight)
        segs.append(w.squeeze(0).cpu().numpy())
    result = _join_natural(segs, chunks, sr, rhythm)
    return torch.from_numpy(result.astype(np.float32)).unsqueeze(0)


# ======================================================================
#  VOICE SAMPLE: isolated voice + the speaker's rhythm (see voiceprep.py)
# ======================================================================

_voice_cache = {}
_voice_lock = threading.Lock()

def _voice_source(speaker_wav: str, key: str = None) -> tuple:
    """The voice sample as every engine uses it: 24 kHz, the voice isolated from
    background music/noise when there is any, plus the speaker's own pause lengths.
    Computed once per voice (cached in memory and in REF_CACHE_DIR).
    Returns (path, rhythm, note)."""
    import hashlib, librosa, soundfile as sf
    if not key:
        with open(speaker_wav, "rb") as f:
            key = "file:" + hashlib.sha1(f.read()).hexdigest()
    hit = _voice_cache.get(key)
    if hit and os.path.exists(hit[0]):
        return hit
    with _voice_lock:
        hit = _voice_cache.get(key)
        if hit and os.path.exists(hit[0]):
            return hit
        REF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tag = f"{key}|voice-v1|{vp.ISOLATE}"
        path = str(REF_CACHE_DIR / (hashlib.sha1(tag.encode()).hexdigest() + "_voice.wav"))
        if os.path.exists(path):
            s, _ = librosa.load(path, sr=INDIC_SR, mono=True)
            note = "prepared earlier"
        else:
            s, _ = librosa.load(speaker_wav, sr=INDIC_SR, mono=True)
            s, note = vp.clean_voice(s, INDIC_SR)
            sf.write(path, s, INDIC_SR)
        rhythm = vp.speaker_rhythm(s, INDIC_SR)
        logger.info("Voice sample: %s; speaker's pauses %.2fs short / %.2fs long (%d found)",
                    note, rhythm["comma"], rhythm["sentence"], rhythm["pauses"])
        res = (path, rhythm, note)
        _voice_cache[key] = res
        return res

def _speech_units(t: str) -> int:
    """Letters + combining vowel signs: a script-independent measure of how much is said.
    (IndicF5 itself measures UTF-8 *bytes*: 1 per English letter but 3 per Tamil letter.)"""
    import unicodedata
    return sum(1 for c in t if unicodedata.category(c)[0] in "LM")

def _is_tamil(t: str) -> bool:
    letters = [c for c in t if c.isalpha()]
    return bool(letters) and sum("஀" <= c <= "௿" for c in letters) / len(letters) > 0.5

_ref_cache = {}

def _indic_reference(speaker_wav: str, key: str = None) -> tuple:
    """Turn the voice sample into an F5-friendly prompt (F5-TTS README: keep it
    under ~12 s and leave silence at the end, otherwise words get truncated):
      * pauses longer than 0.3 s are shortened (the clone copies long pauses, and
        they skew the speaking-rate estimate),
      * clipped to <= INDIC_REF_MAX_S at a pause, never mid-word,
      * a short INDIC_REF_PAD_S of silence appended, so the new speech isn't glued
        to the sample's last word (longer silence made the model waste time).
    Returns (path, speech_seconds, was_clipped); speech_seconds excludes pauses, so
    it measures how fast the person actually talks.  The transcript is always made
    from THIS clip, so the two can never disagree."""
    import hashlib, librosa, soundfile as sf
    if key and key in _ref_cache and os.path.exists(_ref_cache[key][0]):
        return _ref_cache[key]
    vpath, _, _ = _voice_source(speaker_wav, key)      # voice isolated from background
    s, _ = librosa.load(vpath, sr=INDIC_SR, mono=True)
    s = _tidy_pauses(s, INDIC_SR, max_pause=0.3, head=0.05, tail=0.15)
    clipped = len(s) > int(INDIC_REF_MAX_S * INDIC_SR)
    if clipped:   # the cleanest <= 12 s stretch, starting and ending inside pauses
        s, a, b = vp.pick_window(s, INDIC_SR, INDIC_REF_MAX_S)
        s = _tidy_pauses(s, INDIC_SR, max_pause=0.3, head=0.05, tail=0.15)
        logger.info("Voice prompt: using %.1f-%.1fs of the (tidied) sample", a, b)
    speech_s = vp.speaker_rhythm(s, INDIC_SR)["speech_s"] or len(s) / INDIC_SR
    s = np.concatenate([s, np.zeros(int(INDIC_REF_PAD_S * INDIC_SR), np.float32)])
    REF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = str(REF_CACHE_DIR / (hashlib.sha1((key or speaker_wav).encode()).hexdigest() + ".wav"))
    sf.write(path, s, INDIC_SR)
    logger.info("Voice prompt: %.1fs, %.1fs of it speech%s", len(s) / INDIC_SR, speech_s,
                " (clipped)" if clipped else "")
    res = (path, speech_s, clipped)
    if key:
        _ref_cache[key] = res
    return res


def _choose_transcript(typed: str, ref_path: str, key: str, profile_name: str,
                       clipped: bool) -> tuple:
    """A transcript that doesn't match the clip is THE cause of swallowed words
    (F5 'spends' the first new words inside the sample's time slot, which is then
    cut away).  So a typed transcript is only used when it agrees with what the
    recogniser hears; otherwise the detected words are used.  Returns (text, replaced)."""
    _, heard = _auto_transcript(ref_path, (key or ref_path) + PROMPT_TAG, profile_name)
    typed = (typed or "").strip()
    if not typed:
        return heard, False
    diff = asr.cer(typed, heard)
    if not clipped and diff <= 0.35:
        return typed, False
    logger.warning("Typed transcript does not match the voice sample (%s) -- using the detected "
                   "words", "sample was clipped" if clipped else f"{min(diff, 1) * 100:.0f}% different")
    return heard, True


def _indic_generate(model, text: str, ref_path: str, ref_text: str, speech_s: float,
                    rhythm: dict = None, pace: float = 1.0) -> np.ndarray:
    """IndicF5 inference on a prepared prompt (see _indic_reference).
    * Output length from speech units, not UTF-8 bytes (bytes made English-voice
      prompts ask for ~3x too much audio -> babble).
    * Timing like a person reading: the speaker's own speaking rate (pauses excluded)
      times a reading pace, PLUS time for a pause at every comma and sentence end, in
      the speaker's own pause lengths.  Before, the pauses had to fit into the speech
      time, so the words were squeezed together (rushed).
    * The prompt is used as prepared: F5's own preprocessing would strip the
      trailing silence again and could clip audio but not its transcript.
    * Silence clean-up never deletes speech (relative threshold, pauses shortened)."""
    from f5_tts.infer.utils_infer import infer_batch_process
    r = rhythm or vp.DEFAULT_RHYTHM
    ref_wav, ref_sr = torchaudio.load(ref_path)
    ref_s = ref_wav.shape[-1] / ref_sr
    rate = INDIC_DEFAULT_RATE
    if _is_tamil(ref_text) and speech_s > 1:
        rate = min(max(_speech_units(ref_text) / speech_s, INDIC_RATE_MIN), INDIC_RATE_MAX)
    speak = max(rate * pace, 8.0)
    n_sent, n_comma = vp.count_breaks(text)
    pauses = n_sent * r["sentence"] + n_comma * r["comma"]
    units = _speech_units(text)
    gen_s = units / speak + pauses + 0.4 + INDIC_PAUSE_MARK_S
    logger.info("IndicF5: prompt %.1fs, %d units -> %.1fs (speaker %.1f units/s x pace %.2f, "
                "%d+%d pauses = %.1fs)", ref_s, units, gen_s, rate, pace, n_sent, n_comma, pauses)
    audio, _, _ = infer_batch_process(
        (ref_wav, ref_sr), ref_text, [text], model.ema_model, model.vocoder,
        mel_spec_type="vocos", nfe_step=INDIC_NFE_STEPS,
        fix_duration=ref_s + gen_s, device=model.device,
    )
    return _tidy_pauses(np.asarray(audio, np.float32), INDIC_SR,
                        max_pause=max(0.45, r["sentence"] + 0.1))


def synthesise_indic(text: str, ref_path: str, ref_text: str, speech_s: float,
                     rhythm: dict = None, pace: float = 1.0) -> torch.Tensor:
    """IndicF5 zero-shot cloning. Returns (1, N) float tensor at INDIC_SR."""
    model  = get_indic_model()
    ref_t  = _norm_ref_text(ref_text)
    chunks = _split_text(text, INDIC_CHUNK_CHARS)
    logger.info("IndicF5: synthesising %d chunks", len(chunks))
    segs = []
    for i, chunk in enumerate(chunks):
        logger.info("  Chunk %d/%d: '%s...'", i + 1, len(chunks), chunk[:60])
        with torch.inference_mode():
            segs.append(_indic_generate(model, chunk, ref_path, ref_t, speech_s, rhythm, pace))
    result = _join_natural(segs, chunks, INDIC_SR, rhythm)
    return torch.from_numpy(result.astype(np.float32)).unsqueeze(0)


# ======================================================================
#  VOICE-TO-VOICE ("performance"): how the late singers were brought back
#  for new songs -- a living singer performs the melody and rhythm, and the
#  AI only replaces the voice.  Here the user reads the line in their own
#  voice; their exact timing, pauses and emphasis come out in the cloned
#  voice.  Uses Chatterbox's speech tokenizer + voice decoder (S3Gen), the
#  same parts Chatterbox's own voice-conversion model uses; the decoder is
#  already loaded for English, so this needs no extra memory.
# ======================================================================

def _split_at_pauses(s: np.ndarray, sr: int, max_s: float = 20.0) -> list:
    """Pieces of at most max_s seconds, cut in the middle of pauses where possible
    (the speech tokenizer handles 30 s at most)."""
    lim = int(max_s * sr)
    if len(s) <= lim:
        return [s]
    m, hop = vp.speech_mask(s, sr)
    mids = [((a + b) // 2) * hop for a, b, v in vp._runs(m) if not v and a > 0]
    pieces, start = [], 0
    while len(s) - start > lim:
        ok = [c for c in mids if start + 2 * sr < c <= start + lim]
        cut = ok[-1] if ok else start + lim
        pieces.append(s[start:cut])
        start = cut
    pieces.append(s[start:])
    return [p for p in pieces if len(p) > sr // 10]


def perform_voice(guide: np.ndarray, guide_sr: int) -> np.ndarray:
    """Convert a spoken performance into the voice in cb_model.conds (call
    _prepare_voice first).  Returns audio at cb_model.sr."""
    from chatterbox.models.s3tokenizer import S3_SR
    from chatterbox.models.s3gen.const import S3GEN_SIL
    g = torchaudio.functional.resample(torch.from_numpy(np.asarray(guide, np.float32))[None],
                                       guide_sr, S3_SR)[0].numpy()
    steps = 2 if cb_engine == "turbo" else None
    pad = np.zeros(int(0.25 * S3_SR), np.float32)
    # Tested: without a lead-in the first word ("Hey, it's") was lost, and without
    # trailing silence tokens the last word was (Chatterbox's own TTS appends them too).
    sil = torch.full((1, 3), S3GEN_SIL, dtype=torch.long)
    out = []
    for piece in _split_at_pauses(g, S3_SR):
        piece = np.concatenate([pad, piece, pad])
        with torch.inference_mode():
            toks, _ = cb_model.s3gen.tokenizer(torch.from_numpy(piece)[None])
            toks = torch.cat([toks.long().cpu(), sil], dim=1).to(cb_model.device)
            w, _ = cb_model.s3gen.inference(speech_tokens=toks, ref_dict=cb_model.conds.gen,
                                            n_cfm_timesteps=steps)
        w = w.squeeze(0).cpu().numpy().astype(np.float32)
        w = _tidy_pauses(w, cb_model.sr, max_pause=60.0, head=0.1, tail=0.15)   # the padding
        f = min(int(cb_model.sr * 0.005), len(w) // 2)
        if f:
            w[:f] *= np.linspace(0, 1, f, dtype=np.float32)
            w[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
        out.append(w)
    wav = np.concatenate(out)
    # Same inaudible watermark Chatterbox puts on everything it generates.
    return np.asarray(cb_model.watermarker.apply_watermark(wav, sample_rate=cb_model.sr), np.float32)


# ======================================================================
#  API ENDPOINTS
# ======================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": cb_model is not None,
        "en_engine": cb_engine,
        "indic_steps": INDIC_NFE_STEPS,
        "threads": torch.get_num_threads(),
        "indic_loaded": indic_model is not None,
        "indic_error": indic_error,
        "device": DEVICE,
        "ffmpeg": FFMPEG is not None,
        "sample_rate": MODEL_SR,
        "output_sr": OUTPUT_SR,
        "version": "v10-rhythm",
        "languages": {"en": "English", **INDIC_LANGS},
        "perform": cb_model is not None,     # voice-to-voice (/perform)
        "export_formats": list(EXPORT_FORMATS) if FFMPEG else ["wav"],
        "isolate": vp.ISOLATE,
        "tamil_pace": INDIC_PACE,
        "defaults": {
            "exaggeration": DEFAULT_EXAGGERATION,
            "cfg_weight": DEFAULT_CFG_WEIGHT,
        }
    }


def _apply_speed(processed: np.ndarray, out_sr: int, speed: float) -> np.ndarray:
    """Tempo change without pitch shift via ffmpeg atempo (sox is not in the image)."""
    if abs(speed - 1.0) <= 0.02 or not FFMPEG:
        return processed
    src = dst = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f: src = f.name
        dst = src + "_tempo.wav"
        wavfile.write(src, out_sr, (np.clip(processed, -1, 1) * 32767).astype(np.int16))
        r = subprocess.run([FFMPEG, "-y", "-i", src, "-filter:a", f"atempo={speed:.2f}", dst],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        if r.returncode == 0:
            _, d = wavfile.read(dst)
            return (d.astype(np.float32) / 32768.0) if d.dtype == np.int16 else d.astype(np.float32)
    except Exception as e:
        logger.warning("atempo: %s", e)
    finally:
        for p in (src, dst):
            if p:
                try: os.unlink(p)
                except OSError: pass
    return processed


def _auto_transcript(speaker_wav: str, voice_key: str = None, profile_name: str = None) -> tuple:
    """Transcribe the voice prompt (see asr.py).  For profiles the result is stored
    in the profile json, so it survives restarts and runs only once per voice."""
    meta = _profile_meta(profile_name) if profile_name else {}
    saved = meta.get("auto_ref") or {}
    if voice_key and saved.get("key") == voice_key and saved.get("text"):
        return saved.get("lang", "ta"), saved["text"]
    try:
        lang, txt = asr.transcribe(speaker_wav, key=voice_key)
    except Exception as e:
        logger.error("Transcription failed: %s", e, exc_info=True)
        raise HTTPException(503, f"Could not transcribe the voice sample ({e}) -- "
                                 "type its exact words in the transcript box")
    if not txt.strip():
        raise HTTPException(400, "No speech found in the voice sample")
    if profile_name and voice_key:
        meta["auto_ref"] = {"key": voice_key, "lang": lang, "text": txt}
        (PROFILE_DIR / f"{profile_name}.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return lang, txt


@app.post("/transcribe")
def transcribe_sample(voice_sample: UploadFile = File(None), profile_name: str = Form(None)):
    """What is said in the voice sample (for the transcript box)."""
    tmp = None
    try:
        if profile_name:
            profile_name = _safe_profile(profile_name)
            path = PROFILE_DIR / f"{profile_name}.wav"
            if not path.exists():
                raise HTTPException(404, f"Profile '{profile_name}' not found")
            key = f"profile:{profile_name}:{path.stat().st_mtime_ns}"
            path = str(path)
        elif voice_sample:
            import hashlib
            raw = voice_sample.file.read()
            key = "upload:" + hashlib.sha1(raw).hexdigest()
            try:
                path, _ = prepare_audio(raw, voice_sample.filename or "upload.wav", crop=False)
            except RuntimeError as e:
                raise HTTPException(400, str(e))
            tmp = path
        else:
            raise HTTPException(400, "Provide a voice sample or select a saved profile")
        # Same prepared prompt the cloner uses, so the words shown match it exactly.
        ref_path, _, _ = _indic_reference(path, key)
        lang, txt = _auto_transcript(ref_path, key + PROMPT_TAG, profile_name)
        return {"language": lang, "text": txt}
    finally:
        if tmp:
            try: os.unlink(tmp)
            except OSError: pass


def _resolve_voice(voice_sample, profile_name, crop: bool) -> tuple:
    """The voice to clone: a saved profile or an uploaded sample.
    Returns (speaker_wav, voice_key, ref_info, temp_file_or_None, profile_meta)."""
    if profile_name:
        profile_name = _safe_profile(profile_name)
        sp_path = PROFILE_DIR / f"{profile_name}.wav"
        if not sp_path.exists():
            raise HTTPException(404, f"Profile '{profile_name}' not found")
        meta = _profile_meta(profile_name)
        ref_info = analyse_reference(str(sp_path))
        ref_info["duration"] = meta.get("duration", 99)
        return (str(sp_path), f"profile:{profile_name}:{sp_path.stat().st_mtime_ns}",
                ref_info, None, meta)
    if voice_sample:
        import hashlib
        raw = voice_sample.file.read()
        try:
            speaker_wav, ref_info = prepare_audio(raw, voice_sample.filename or "upload.wav", crop=crop)
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        return (speaker_wav, "upload:" + hashlib.sha1(raw).hexdigest(), ref_info, speaker_wav,
                {"_raw": raw, "_fname": voice_sample.filename or "upload.wav"})
    raise HTTPException(400, "Provide a voice sample or select a saved profile")


@app.post("/perform")
def perform(
    request:      Request,
    guide_audio:  UploadFile = File(...),     # the user reading the line, any language
    voice_sample: UploadFile = File(None),
    profile_name: str        = Form(None),
):
    """Voice-to-voice: your own reading -- its rhythm, pauses and emphasis -- comes out
    in the cloned voice (see perform_voice).  Takes seconds, not minutes."""
    import librosa
    if cb_model is None:
        raise HTTPException(503, "Model loading -- please wait")
    tmp_speaker = guide_raw = guide_wav = None
    out_path = OUTPUT_DIR / f"{uuid.uuid4().hex}.wav"
    try:
        speaker_wav, voice_key, _, tmp_speaker, vmeta = _resolve_voice(voice_sample, profile_name, crop=True)
        data = guide_audio.file.read()
        ext = Path(guide_audio.filename or "guide.wav").suffix.lower() or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            f.write(data); guide_raw = f.name
        guide_wav = guide_raw + "_g.wav"
        if not (_ffmpeg_convert(guide_raw, guide_wav) or
                (ext not in _NEEDS_FFMPEG and _torchaudio_convert(guide_raw, guide_wav))):
            raise HTTPException(400, f"Cannot read the recording ('{ext}'). Install ffmpeg.")
        g, gsr = librosa.load(guide_wav, sr=INDIC_SR, mono=True)
        g_s = len(g) / gsr
        if not (0.5 <= g_s <= PERFORM_MAX_S):
            raise HTTPException(400, f"Record between 0.5 and {PERFORM_MAX_S}s (yours is {g_s:.1f}s)")
        g, guide_note = vp.clean_voice(g, gsr)          # background out of the performance too
        g = _tidy_pauses(g, gsr, max_pause=60.0, head=0.1, tail=0.2)   # edges only
        with SYNTH_LOCK:
            voice_path, _, prep_note = _voice_source(speaker_wav, voice_key)
            _prepare_voice(voice_path, DEFAULT_EXAGGERATION, voice_key)
            logger.info("Performance: %.1fs recording (%s) -> cloned voice", g_s, guide_note)
            wav = perform_voice(g, gsr)
            raw_wav_path = str(out_path) + "_raw.wav"
            torchaudio.save(raw_wav_path, torch.from_numpy(wav).unsqueeze(0), cb_model.sr)
            processed, out_sr = professional_chain(raw_wav_path, cb_model.sr, {"rms": 0.035, "hf_lf_ratio": 0.11}, engine="chatterbox")
        torchaudio.save(str(out_path), torch.from_numpy(processed).unsqueeze(0), out_sr)
        try: os.unlink(raw_wav_path)
        except OSError: pass
        logger.info("Done [perform]: %.1fs @ %dHz", len(processed) / out_sr, out_sr)
        activity.record("perform", activity.user_id(request), data, guide_audio.filename or "guide.wav",
                        str(out_path), {"voice": f"profile:{profile_name}" if profile_name else
                                        f"upload:{vmeta.get('_fname', '')}", "seconds": round(len(processed) / out_sr, 2)})
        return FileResponse(str(out_path), media_type="audio/wav", filename="performed_voice.wav",
                            headers={"X-Voice-Prep": _quote(prep_note)})
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Perform error: %s", e, exc_info=True)
        raise HTTPException(500, f"Voice-to-voice failed: {e}")
    finally:
        for p in (tmp_speaker, guide_raw, guide_wav):
            if p:
                try: os.unlink(p)
                except OSError: pass


# Plain `def` (not async): FastAPI runs it in a worker thread, so a long
# synthesis no longer freezes /health and other requests.
@app.post("/clone")
def clone(
    request:      Request,
    text:         str        = Form(...),
    language:     str        = Form("en"),
    speed:        float      = Form(1.0),
    exaggeration: float      = Form(DEFAULT_EXAGGERATION),
    cfg_weight:   float      = Form(DEFAULT_CFG_WEIGHT),
    voice_sample: UploadFile = File(None),
    profile_name: str        = Form(None),
    ref_text:     str        = Form(None),
    style:        str        = Form("spoken"),   # Tamil: "spoken" (everyday) | "formal" (as typed)
):
    text = (text or "").strip()
    if not text:
        raise HTTPException(400, "Text cannot be empty")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(400, f"Text too long ({len(text)} chars, max {MAX_TEXT_CHARS})")

    language = (language or "en").strip().lower()
    is_indic = language in INDIC_LANGS
    if language != "en" and not is_indic:
        raise HTTPException(400, f"Unsupported language '{language}'")
    if language == "en" and _is_tamil(text):
        # The English engine can't read Tamil script: it rambles until its 1000-token
        # cap (~40s of noise). Refuse instead of returning garbage.
        raise HTTPException(400, "This text is in Tamil -- select Tamil as the language")
    if not is_indic and cb_model is None:
        raise HTTPException(503, "Model loading -- please wait")

    exaggeration = max(0.1, min(float(exaggeration), 1.5))
    cfg_weight   = max(0.0, min(float(cfg_weight), 1.0))
    speed        = max(0.5, min(float(speed), 2.0))

    tmp_speaker = None
    out_path    = OUTPUT_DIR / f"{uuid.uuid4().hex}.wav"

    try:
        # Indic: keep the whole clip (it is clipped later, together with its transcript).
        speaker_wav, voice_key, ref_info, tmp_speaker, meta = _resolve_voice(
            voice_sample, profile_name, crop=not is_indic)
        if profile_name:
            profile_name = _safe_profile(profile_name)
            if is_indic and not (ref_text or "").strip():
                ref_text = meta.get("ref_text", "")
        if is_indic:
            dur = ref_info.get("duration", 0)
            if isinstance(dur, (int, float)) and dur != 99 and not (INDIC_MIN_REF_S <= dur <= INDIC_MAX_REF_S):
                raise HTTPException(400, f"For {INDIC_LANGS[language]} use a {INDIC_MIN_REF_S}-"
                                         f"{INDIC_MAX_REF_S}s voice sample (yours is {dur:.1f}s)")
        # Voice isolated from background music/noise + the speaker's own pause lengths.
        voice_path, rhythm, prep_note = _voice_source(speaker_wav, voice_key)

        spoken_text = None
        if is_indic:
            # Prepared prompt + a transcript verified against it (wrong one = lost words).
            ref_path, speech_s, clipped = _indic_reference(speaker_wav, voice_key)
            ref_text, ref_replaced = _choose_transcript(ref_text, ref_path, voice_key,
                                                        profile_name, clipped)
            spoken_text = prepare_for_tts(text, spoken=(style or "spoken").lower() != "formal")
            logger.info("Tamil text (%s): %s", style, spoken_text)

        with SYNTH_LOCK:
            if is_indic:
                try:
                    # Tamil speed is applied while generating (natural), not stretched after.
                    wav = synthesise_indic(spoken_text, ref_path, ref_text, speech_s,
                                           rhythm, pace=INDIC_PACE * speed)
                except HTTPException:
                    raise
                except Exception as e:
                    logger.error("IndicF5 error: %s", e, exc_info=True)
                    raise HTTPException(503, f"{INDIC_LANGS[language]} engine unavailable: {e}")
                src_sr, engine = INDIC_SR, "indic"
            else:
                src_sr = MODEL_SR or CHATTERBOX_SR
                wav    = synthesise(text, voice_path, exaggeration, cfg_weight, src_sr,
                                    voice_key=voice_key, rhythm=rhythm)
                engine = "chatterbox"

            raw_wav_path = str(out_path) + "_raw.wav"
            torchaudio.save(raw_wav_path, wav.cpu(), src_sr)
            processed, out_sr = professional_chain(raw_wav_path, src_sr, ref_info, engine=engine)
            if not is_indic:
                processed = _apply_speed(processed, out_sr, speed)

        torchaudio.save(str(out_path), torch.from_numpy(processed).unsqueeze(0), out_sr)
        try: os.unlink(raw_wav_path)
        except OSError: pass

        logger.info("Done [%s/%s]: %.1fs @ %dHz", engine, language, len(processed)/out_sr, out_sr)
        activity.record("clone", activity.user_id(request),
                        meta.get("_raw") or speaker_wav,
                        meta.get("_fname") or f"profile-{profile_name}.wav", str(out_path),
                        {"language": language, "text": text, "spoken_text": spoken_text or text,
                         "voice": f"profile:{profile_name}" if profile_name else "upload",
                         "seconds": round(len(processed) / out_sr, 2), "speed": speed})

        headers = {"X-Voice-Prep": _quote(prep_note),
                   "X-Rhythm": f"short={rhythm['comma']:.2f};long={rhythm['sentence']:.2f}"}
        dur = ref_info.get("duration", 99)
        if not is_indic and isinstance(dur, (int, float)) and dur < WARN_DURATION_S:
            headers["X-Quality-Warning"] = f"Reference {dur:.1f}s -- use 15-30s for best results"
        if is_indic:   # let the page show what was actually spoken
            headers["X-Spoken-Text"] = _quote(spoken_text)
            headers["X-Ref-Text"] = _quote(ref_text.strip())
            if ref_replaced:
                headers["X-Ref-Replaced"] = "1"

        return FileResponse(str(out_path), media_type="audio/wav",
                            filename="cloned_voice.wav", headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Clone error: %s", e, exc_info=True)
        raise HTTPException(500, f"Synthesis failed: {e}")
    finally:
        if tmp_speaker:
            try: os.unlink(tmp_speaker)
            except OSError: pass


# Download formats for /export: ffmpeg encoder settings + content type.
EXPORT_FORMATS = {
    "mp3":  (["-c:a", "libmp3lame", "-b:a", "192k"], "audio/mpeg"),
    "wav":  (["-c:a", "pcm_s16le"], "audio/wav"),
    "flac": (["-c:a", "flac"], "audio/flac"),
    "m4a":  (["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"], "audio/mp4"),
    "ogg":  (["-c:a", "libvorbis", "-q:a", "6"], "audio/ogg"),
    "opus": (["-c:a", "libopus", "-b:a", "96k"], "audio/ogg"),
    "aiff": (["-c:a", "pcm_s16be"], "audio/aiff"),
    "wma":  (["-c:a", "wmav2", "-b:a", "192k"], "audio/x-ms-wma"),
}
EXPORT_MAX_BYTES = 60 * 1024 * 1024


@app.post("/export")
def export_audio(audio: UploadFile = File(...), format: str = Form("mp3")):
    """Convert a generated clip (the WAV that /clone or /perform returned) into the
    format picked in the download dialog.  Only WAV input is accepted, and ffmpeg is
    told so (-f wav): it never follows playlists or other files named in an upload."""
    fmt = (format or "").strip().lower()
    if fmt not in EXPORT_FORMATS:
        raise HTTPException(400, f"Unknown format '{format}' -- choose one of: {', '.join(EXPORT_FORMATS)}")
    if not FFMPEG:
        raise HTTPException(503, "Converting needs ffmpeg on the server -- download as WAV instead")
    data = audio.file.read(EXPORT_MAX_BYTES + 1)
    if len(data) > EXPORT_MAX_BYTES:
        raise HTTPException(413, "Audio too large to convert")
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise HTTPException(400, "Expected the WAV file produced by the cloner")
    opts, mime = EXPORT_FORMATS[fmt]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(data); src = f.name
    dst = src[:-4] + "_export." + fmt
    try:
        r = subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-f", "wav", "-i", src,
                            "-vn", "-map_metadata", "-1", *opts, dst],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120)
    finally:
        try: os.unlink(src)
        except OSError: pass
    if r.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
        try: os.unlink(dst)
        except OSError: pass
        logger.error("Export to %s failed: %s", fmt, r.stderr.decode(errors="ignore")[-300:])
        raise HTTPException(500, f"Could not convert to {fmt.upper()}")
    logger.info("Export: %s, %.0f KB", fmt, os.path.getsize(dst) / 1024)
    return FileResponse(dst, media_type=mime, filename=f"cloned_voice.{fmt}",
                        background=BackgroundTask(os.unlink, dst))


@app.post("/save-profile")
def save_profile(
    profile_name: str        = Form(...),
    voice_sample: UploadFile = File(...),
    ref_text:     str        = Form(None),
    language:     str        = Form("en"),
):
    profile_name = _safe_profile(profile_name)
    language     = (language or "en").strip().lower()
    is_indic     = language in INDIC_LANGS
    raw   = voice_sample.file.read()
    fname = voice_sample.filename or "upload.wav"
    try:
        # A profile with a transcript must keep its full clip to stay in sync.
        tmp, info = prepare_audio(raw, fname, crop=not (is_indic and (ref_text or "").strip()))
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    dest = PROFILE_DIR / f"{profile_name}.wav"
    os.replace(tmp, dest)
    meta = {"duration": round(info.get("duration", 0), 2),
            "ref_text": (ref_text or "").strip(),
            "language": language}
    (PROFILE_DIR / f"{profile_name}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return {
        "status": "saved",
        "profile": profile_name,
        "duration": round(info.get("duration", 0), 1),
        "has_transcript": bool(meta["ref_text"]),
        "rms_db": round(20 * np.log10(info.get("rms", 0.035) + 1e-9), 1),
    }

@app.get("/profiles")
async def list_profiles():
    names = sorted(f.stem for f in PROFILE_DIR.glob("*.wav"))
    details = []
    for n in names:
        m = _profile_meta(n)
        details.append({"name": n, "language": m.get("language", "en"),
                        "has_transcript": bool(m.get("ref_text"))})
    return {"profiles": names, "details": details}

@app.get("/profiles/{name}/audio")
async def profile_audio(name: str):
    """The saved voice sample, so the page can play it before using the profile."""
    name = _safe_profile(name)
    p = PROFILE_DIR / f"{name}.wav"
    if not p.exists():
        raise HTTPException(404, f"Profile '{name}' not found")
    return FileResponse(str(p), media_type="audio/wav", headers={"Cache-Control": "no-cache"})

@app.delete("/profiles/{name}")
async def delete_profile(name: str):
    name = _safe_profile(name)
    for ext in (".wav", ".json"):
        p = PROFILE_DIR / f"{name}{ext}"
        if p.exists(): p.unlink()
    return {"status": "deleted", "profile": name}


if __name__ == "__main__":
    # PORT env var support: HF Spaces uses 7860, local default is 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)