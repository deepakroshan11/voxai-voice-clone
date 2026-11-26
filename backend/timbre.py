"""Voice colour matching: give a Tamil clone the speaker's own spectral balance (body,
warmth, presence, air) -- the "Match EQ" technique audio engineers use, applied per speaker.

Why: what makes a voice recognisable is mainly its pitch and its spectral envelope
(formants, body, brightness); in listening tests, changing those hurt speaker recognition
the most (Sell & Suied 2015; Kitamura & Akagi).  F5-TTS (the Tamil engine) copies the
colour of its prompt only approximately -- e.g. a female clone's 6-10 kHz "air" was 8.4 dB
low (dull) and its 3-5 kHz a little hot.

Tested on real clones (see the README):
  * learned on part of a speaker's sample, the EQ brought the clone's spectrum closer to
    the speaker's OTHER, unseen speech (2.5 -> 1.8, 1.8 -> 1.4, 2.3 -> 1.5 dB), i.e. to
    within the voice's own natural variation -- it captures the voice, not the words;
  * words stay as clear (speech recognition unchanged).
  (Moving the clone's pitch level to the speaker's, and re-voicing with S3Gen voice
  conversion, were also tested and dropped: both lowered similarity for some voices.)

Method (Match EQ, as in Logic/FabFilter/Ozone):
  * long-term average spectrum (LTAS) of the speaker's voice sample and of the clone, over
    speech frames only, in 1/3-octave bands (~4 semitones of smoothing, the usual setting);
  * the difference, smoothed once more across neighbouring bands, scaled to AMOUNT and
    capped (no more than +LIMIT_UP / -LIMIT_DOWN dB; never a boost below 0.7x the
    speaker's pitch, where a sample holds only room noise, and at most +5 dB of air, so
    hum, rumble or hiss in a sample are never copied);
  * applied as one linear-phase FIR filter; the 500-2000 Hz level stays where it was.
"""
import logging

import numpy as np

logger = logging.getLogger("voxai")

CENTERS = np.array([63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000,
                    2500, 3150, 4000, 5000, 6300, 8000, 10000], float)
AMOUNT = 0.75          # share of the measured difference that is corrected
LIMIT_UP, LIMIT_DOWN = 8.0, 6.0
LOW_CAP = 3.0          # 63-80 Hz: at most +3 dB (no rumble)
AIR_CAP = 5.0          # 6.3-10 kHz: at most +5 dB (no hiss)
MIN_SPEECH_S = 1.5


def ltas(y: np.ndarray, sr: int):
    """1/3-octave long-term average spectrum (dB) of the speech frames, 0 dB = 500-2000 Hz.
    None if there is too little speech."""
    import librosa
    y = np.asarray(y, np.float32)
    if len(y) < sr * MIN_SPEECH_S:
        return None
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256)) ** 2
    e = 10 * np.log10(S.sum(0) + 1e-12)
    keep = e > np.percentile(e, 90) - 30
    if keep.sum() * 256 / sr < MIN_SPEECH_S:
        return None
    P = S[:, keep].mean(1)
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    out = []
    for c in CENTERS:
        sel = (f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))
        out.append(10 * np.log10(P[sel].sum() + 1e-12) if sel.any() and c < sr / 2 * 0.95 else np.nan)
    out = np.array(out)
    mid = out[(CENTERS >= 500) & (CENTERS <= 2000)]
    return out - np.nanmean(mid)


def correction(speaker_ltas, clone_ltas, speaker_hz=None):
    """The capped, smoothed EQ curve (dB per band) that moves the clone toward the speaker."""
    d = np.asarray(speaker_ltas, float) - np.asarray(clone_ltas, float)
    d = np.where(np.isfinite(d), d, 0.0)
    d = np.convolve(np.pad(d, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")   # smooth across bands
    d = np.clip(d * AMOUNT, -LIMIT_DOWN, LIMIT_UP)
    d[CENTERS < 90] = np.minimum(d[CENTERS < 90], LOW_CAP)
    if speaker_hz:                         # below the voice's own lowest notes: only room noise
        below = CENTERS < 0.7 * speaker_hz
        d[below] = np.minimum(d[below], 0.0)
    d[CENTERS > 6000] = np.minimum(d[CENTERS > 6000], AIR_CAP)
    d -= d[(CENTERS >= 500) & (CENTERS <= 2000)].mean()                            # keep the mids' level
    return d


def apply_curve(y: np.ndarray, sr: int, curve_db: np.ndarray) -> np.ndarray:
    """Linear-phase FIR with the given band gains (no delay in the result)."""
    from scipy import signal
    nyq = sr / 2
    use = CENTERS < nyq * 0.95
    freqs = np.concatenate([[0.0], CENTERS[use], [nyq]])
    gains_db = np.concatenate([[curve_db[use][0] - 6.0], curve_db[use], [curve_db[use][-1] - 3.0]])
    taps = 2047
    h = signal.firwin2(taps, freqs / nyq, 10 ** (gains_db / 20), window="hann")
    out = signal.fftconvolve(np.asarray(y, np.float64), h)[taps // 2: taps // 2 + len(y)]
    return out.astype(np.float32)


def match(y: np.ndarray, sr: int, speaker_ltas, speaker_hz=None):
    """Give the clone the speaker's spectral balance.  Returns (audio, note)."""
    if speaker_ltas is None:
        return y, "colour: no reference measure"
    try:
        mine = ltas(y, sr)
        if mine is None:
            return y, "colour: too little speech"
        curve = correction(speaker_ltas, mine, speaker_hz)
        if np.max(np.abs(curve)) < 1.0:
            return y, "colour: already matched"
        out = apply_curve(y, sr, curve)
        pk = np.abs(out).max() + 1e-9
        if pk > 0.99:
            out *= 0.99 / pk
        grp = lambda lo, hi: float(curve[(CENTERS >= lo) & (CENTERS <= hi)].mean())
        note = (f"colour: bass {grp(63, 160):+.1f}, body {grp(200, 400):+.1f}, presence {grp(1250, 2500):+.1f}, "
                f"clarity {grp(3150, 5000):+.1f}, air {grp(6300, 10000):+.1f} dB")
        logger.info("Voice colour matched to the speaker (%s)", note)
        return out, note
    except Exception as e:
        logger.warning("Voice colour matching skipped: %s", e)
        return y, "colour: skipped"
