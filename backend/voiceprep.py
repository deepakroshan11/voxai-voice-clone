"""Voice-sample preparation and speech rhythm.

Follows how studios recreate a voice that can no longer be recorded (the Beatles'
"Now and Then", A.R. Rahman's "Thimiri Yezhuda", Yuvan's "Chinna Chinna Kangal"):

 1. Isolate the voice. A clone copies everything in the sample, so background music
    or noise ends up in every sentence.  "Now and Then" only became possible once
    Lennon's voice was de-mixed from the piano on the cassette, and RVC/UVR voice
    models are trained on de-mixed vocals.  Here: the vocal stem of Hybrid Demucs
    (built into torchaudio), used only when the sample has a real background.
 2. Use the cleanest stretch of the sample, cut inside pauses, never mid-word.
 3. Keep the person's own timing.  In those songs a living singer performs the
    melody and rhythm and the AI only supplies the voice (see /perform in main.py).
    For text-to-speech, the pauses and speaking rate are measured from the sample.
    Studies of read speech find pauses grow with the strength of the break (word <
    comma < sentence), and pause length differs a lot between speakers (one study
    measured 176 ms against 687 ms at the same break).  So pauses are measured per
    speaker, not fixed.
"""
import gc
import logging
import os
import threading

import numpy as np
import torch
import torchaudio

logger = logging.getLogger("voxai")

FRAME_S = 0.02
MIN_PAUSE_S = 0.12          # shorter silences are stop consonants, not pauses
ISOLATE = os.environ.get("VOXAI_ISOLATE", "auto").strip().lower()   # auto | on | off
ISOLATE_BG_DB = -32.0       # "auto": isolate when the background is louder than this


def frame_db(s: np.ndarray, sr: int, frame_s: float = FRAME_S):
    hop = max(1, int(sr * frame_s))
    n = len(s) // hop
    if n == 0:
        return np.zeros(0), hop
    x = np.asarray(s[:n * hop], np.float64).reshape(n, hop)
    return 20 * np.log10(np.sqrt((x ** 2).mean(1)) + 1e-9), hop


def background_db(s: np.ndarray, sr: int) -> float:
    """Background level relative to the speech, in dB.  A quiet room is around
    -40 dB or lower; film clips with music or noisy phone calls are -30 to -15 dB."""
    db, _ = frame_db(s, sr)
    if len(db) < 10:
        return -60.0
    return float(np.percentile(db, 10) - np.percentile(db, 95))


def _runs(mask: np.ndarray):
    """(start, end, value) runs of a boolean array."""
    if not len(mask):
        return []
    edges = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    b = np.concatenate([[0], edges, [len(mask)]])
    return [(int(b[i]), int(b[i + 1]), bool(mask[b[i]])) for i in range(len(b) - 1)]


def speech_mask(s: np.ndarray, sr: int, floor_db: float = -35.0):
    """Speech frames (relative to the loudest frame), with gaps shorter than
    MIN_PAUSE_S filled in, so a word's stop consonants don't count as pauses."""
    db, hop = frame_db(s, sr)
    if not len(db):
        return np.zeros(0, bool), hop
    m = db > db.max() + floor_db
    short = int(round(MIN_PAUSE_S / FRAME_S))
    for a, b, v in _runs(m):
        if not v and a > 0 and b < len(m) and b - a < short:
            m[a:b] = True
    return m, hop


# ── 1. voice isolation ──────────────────────────────────────────────────────

_sep_lock = threading.Lock()


def isolate_voice(s: np.ndarray, sr: int) -> np.ndarray:
    """Vocal stem of Hybrid Demucs (torchaudio HDEMUCS_HIGH_MUSDB_PLUS, about 320 MB,
    downloaded once).  Loaded for the call and freed afterwards: RAM is tight."""
    from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS as bundle
    with _sep_lock:
        model = bundle.get_model().eval()
        try:
            msr = bundle.sample_rate
            x = torchaudio.functional.resample(torch.from_numpy(np.asarray(s, np.float32))[None], sr, msr)
            x = x.repeat(2, 1)[None]                       # the model is stereo
            mean, std = x.mean(), x.std() + 1e-8
            x = (x - mean) / std
            vi = list(model.sources).index("vocals")
            n = x.shape[-1]
            seg, fade = int(10 * msr), int(0.5 * msr)
            out, wsum = torch.zeros(n), torch.zeros(n)
            st = 0
            while True:                                   # 10 s pieces, 0.5 s cross-fades
                piece = x[..., st:st + seg]
                with torch.inference_mode():
                    y = model(piece)[0, vi].mean(0)
                k = y.shape[-1]
                w = torch.ones(k)
                f = min(fade, k // 2)
                if st > 0:
                    w[:f] = torch.linspace(0, 1, f)
                if st + k < n:
                    w[-f:] = torch.linspace(1, 0, f)
                out[st:st + k] += y * w
                wsum[st:st + k] += w
                if st + k >= n:
                    break
                st += seg - fade
            v = out / wsum.clamp_min(1e-6) * std + mean
            v = torchaudio.functional.resample(v[None], msr, sr)[0].numpy()
        finally:
            del model
            gc.collect()
    return v[:len(s)].astype(np.float32)


def clean_voice(s: np.ndarray, sr: int) -> tuple:
    """Isolate the voice when the sample has a real background (or VOXAI_ISOLATE=on).
    Returns (audio, what_was_done)."""
    if ISOLATE == "off":
        return s, "off"
    bg = background_db(s, sr)
    if ISOLATE != "on" and bg < ISOLATE_BG_DB:
        return s, f"clean sample (background {bg:.0f} dB)"
    try:
        v = isolate_voice(s, sr)
    except Exception as e:                      # no network for the weights, etc.
        logger.warning("Voice isolation failed (%s) -- using the sample as it is", e)
        return s, "isolation failed"
    # Speech that Demucs put in the "other" stem would be lost: keep the original then.
    m, hop = speech_mask(s, sr)
    k = min(len(m) * hop, len(v))
    act = np.repeat(m, hop)[:k]
    if act.any():
        kept = np.sqrt((v[:k][act] ** 2).mean() / ((s[:k][act] ** 2).mean() + 1e-12))
        if kept < 0.45:
            logger.warning("Voice isolation kept only %.0f%% of the speech -- using the sample "
                           "as it is", kept * 100)
            return s, "isolation rejected"
    return v, f"voice isolated (background was {bg:.0f} dB, now {background_db(v, sr):.0f} dB)"


# ── 2. cleanest stretch ─────────────────────────────────────────────────────

def pick_window(s: np.ndarray, sr: int, max_s: float, min_s: float = 4.0) -> tuple:
    """Cleanest stretch of at most max_s seconds, starting and ending inside pauses
    (prefers longer stretches: more of the voice; avoids clipping and background).
    Returns (audio, start_s, end_s)."""
    if len(s) <= max_s * sr:
        return s, 0.0, len(s) / sr
    m, hop = speech_mask(s, sr)
    db, _ = frame_db(s, sr)
    cuts = [0]
    for a, b, v in _runs(m):
        if not v and a > 0 and b < len(m) and b - a >= 4:      # pause >= 80 ms
            cuts.append(((a + b) // 2) * hop)
    cuts.append(len(s))
    best, best_score = None, -1e9
    for i, a in enumerate(cuts):
        for b in cuts[i + 1:]:
            d = (b - a) / sr
            if d > max_s:
                break
            if d < min_s:
                continue
            fa, fb = a // hop, max(a // hop + 1, b // hop)
            seg_db = db[fa:fb]
            bg = float(np.percentile(seg_db, 10) - np.percentile(seg_db, 95)) if len(seg_db) > 10 else -60
            clip = float(np.mean(np.abs(s[a:b]) > 0.98))
            score = d - 0.15 * max(0.0, bg + 45) - 500 * clip
            if score > best_score:
                best, best_score = (a, b), score
    if best is None:                       # no pauses at all: first max_s seconds
        lim = int(max_s * sr)
        return s[:lim], 0.0, lim / sr
    a, b = best
    return s[a:b], a / sr, b / sr


# ── 3. the speaker's rhythm ─────────────────────────────────────────────────

DEFAULT_RHYTHM = {"comma": 0.22, "sentence": 0.45, "pauses": 0, "speech_s": 0.0}


def speaker_rhythm(s: np.ndarray, sr: int) -> dict:
    """Pause lengths and speaking time measured from the sample.
    comma = a typical short pause, sentence = a typical long pause (between phrases),
    speech_s = seconds of actual speech (pauses excluded)."""
    m, hop = speech_mask(s, sr)
    if not m.any():
        return dict(DEFAULT_RHYTHM)
    gaps = [(b - a) * FRAME_S for a, b, v in _runs(m) if not v and a > 0 and b < len(m)]
    speech_s = float(m.sum() * FRAME_S)
    r = dict(DEFAULT_RHYTHM, pauses=len(gaps), speech_s=speech_s)
    if len(gaps) >= 2:
        g = np.asarray(gaps)
        comma = float(np.clip(np.percentile(g, 35), 0.15, 0.45))
        sentence = float(np.clip(np.percentile(g, 85), comma + 0.1, 0.8))
        r.update(comma=round(comma, 2), sentence=round(sentence, 2))
    elif len(gaps) == 1:
        r.update(comma=round(float(np.clip(gaps[0] * 0.6, 0.15, 0.45)), 2),
                 sentence=round(float(np.clip(gaps[0], 0.3, 0.8)), 2))
    return r


def count_breaks(text: str) -> tuple:
    """(sentence breaks, comma breaks) inside a piece of text (not at its end)."""
    import re
    t = text.strip()
    sentences = len(re.findall(r"[.!?।]+[\"'”’)]*\s+(?=\S)", t))
    commas = len(re.findall(r"[,;:—–]\s+(?=\S)", t))
    return sentences, commas
