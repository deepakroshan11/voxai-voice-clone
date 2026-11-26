"""Tamil pacing: an even, natural talking speed, and pauses only where a speaker makes them.

What went wrong (measured on real clones, 26 Nov 2026): IndicF5 gets ONE length for the
whole text and decides by itself how to spend it.  It spent it on long gaps at punctuation
-- 0.5-1 s, filled with the voice sample's background hiss -- and then crammed the rest:
phrase speeds of 3.9 / 10.3 / 7.8 syllables per second inside one 5-second clip, with only
58% of the syllables still distinct.  Natural Tamil is about 131 words/min read aloud and
149 in a monologue (Rakesh et al. 2016, Language in India 16:8; 25 native speakers), i.e.
roughly 5.5-6.5 syllables per second.

What this module does (main.synthesise_indic uses it):
  * pieces(): one IndicF5 pass per sentence, each with its own time budget -- the model
    can't borrow time from one sentence to pause somewhere else (F5-TTS's own
    infer_process also generates long text in short chunks).  Tested: the sentences then
    come out at the planned speed (planned 5.0 -> 4.9 and 4.6 syllables/s).
    Very short sentences ("ஓ!", "சரி.") ride along with the next one, and a comma glued
    to the next word ("a,b") is spaced, so it reads as the pause it is.
  * join(): each piece trimmed to its speech (found by voicing, so hiss or a breath at the
    edge isn't kept as "speech"), long gaps inside it shortened, a piece still faster than
    MAX_RATE slowed (pitch kept), and the pieces joined with the speaker's own pause
    lengths kept within natural bounds -- filled with the clip's own quiet room sound,
    not dead digital silence.
"""
import logging
import re

import numpy as np

logger = logging.getLogger("voxai")

SENTENCE_PAUSE = (0.35, 0.60)   # pause between sentences: the speaker's own, kept in this range
COMMA_PAUSE = (0.15, 0.28)      # after a piece that ends at a comma (long sentences are split there)
INNER_MAX = 0.32                # longest pause kept inside a sentence (commas)
MAX_RATE = 6.8                  # syllables/s while talking; a faster piece is slowed ...
MIN_TEMPO = 0.8                 # ... but never below 0.8x
MIN_SYLLABLES = 3               # shorter sentences join the next one
HEAD, TAIL = 0.04, 0.10         # kept before the first / after the last sound of a piece
# Time budget of one pass (main._indic_generate, paced): the words at the planned speed
# + COMMA_ALLOW per comma + END_ALLOW for the end of the sentence (+ the prompt's pause mark).
COMMA_ALLOW = 0.20
END_ALLOW = 0.30
FRAME = 0.01

_VOWELS = {chr(c) for c in range(0x0B85, 0x0B95)}     # அ .. ஔ
_CONSONANTS = {chr(c) for c in range(0x0B95, 0x0BBA)}  # க .. ஹ
_PULLI = "்"                                     # ் (no vowel: not a syllable)


def syllables(text: str) -> int:
    """Tamil syllables: an independent vowel, or a consonant not closed by pulli."""
    n = 0
    for i, c in enumerate(text):
        if c in _VOWELS or (c in _CONSONANTS and not (i + 1 < len(text) and text[i + 1] == _PULLI)):
            n += 1
    if n == 0:   # not Tamil: count vowel groups
        n = len(re.findall(r"[aeiouy]+", text.lower()))
    return n


def spaced(text: str) -> str:
    """'a,b' -> 'a, b' (but not inside numbers like 3.5)."""
    return re.sub(r"([,;:.!?])(?=[^\s\d.,;:!?\"'”’)])", r"\1 ", text)


def pieces(text: str) -> list:
    """One piece per sentence; a sentence of fewer than MIN_SYLLABLES joins the next."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?।])\s+", spaced(text).strip()) if p.strip()]
    out, carry = [], ""
    for i, p in enumerate(parts):
        p = f"{carry} {p}".strip() if carry else p
        carry = ""
        if syllables(p) < MIN_SYLLABLES and i + 1 < len(parts):
            carry = p
            continue
        out.append(p)
    return out


# ── where the speech is ─────────────────────────────────────────────────────

def _frame_db(s: np.ndarray, sr: int) -> np.ndarray:
    hop = int(sr * FRAME)
    n = len(s) // hop
    x = np.asarray(s[:n * hop], np.float64).reshape(n, hop)
    return 20 * np.log10(np.sqrt((x ** 2).mean(1)) + 1e-9)


def _voiced(s: np.ndarray, sr: int) -> np.ndarray:
    """Per 10 ms frame: is there voice (pitch)?  Hiss and breaths have none."""
    import parselmouth
    snd = parselmouth.Sound(np.asarray(s, np.float64), sr)
    p = snd.to_pitch_ac(time_step=FRAME, pitch_floor=60, pitch_ceiling=500)
    f0 = p.selected_array["frequency"]
    n = len(s) // int(sr * FRAME)
    v = np.zeros(n, bool)
    idx = np.clip(np.round((p.xs() - FRAME / 2) / FRAME).astype(int), 0, max(n - 1, 0))
    v[idx[f0 > 0]] = True
    return v


def _runs(mask: np.ndarray):
    if not len(mask):
        return []
    edges = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    b = np.concatenate([[0], edges, [len(mask)]])
    return [(int(b[i]), int(b[i + 1]), bool(mask[b[i]])) for i in range(len(b) - 1)]


def _layout(s: np.ndarray, sr: int):
    """(voiced flags, loud flags, frame dB): loud = clearly above the clip's own floor."""
    db = _frame_db(s, sr)
    v = _voiced(s, sr)[:len(db)]
    floor = np.percentile(db, 10)
    loud = db > max(floor + 14.0, db.max() - 40.0)
    return v, loud, db


def speech_span(s: np.ndarray, sr: int):
    """(start, end) samples of the speech: first to last voiced sound, plus a consonant
    right next to it (a loud unvoiced run of <= 0.2 s) -- not hiss or a breath."""
    v, loud, _ = _layout(s, sr)
    if not v.any():
        return 0, len(s)
    hop = int(sr * FRAME)
    first, last = int(np.flatnonzero(v)[0]), int(np.flatnonzero(v)[-1])
    a = first
    while a > 0 and loud[a - 1] and first - a < 20:
        a -= 1
    b = last + 1
    while b < len(v) and loud[b] and b - last < 20:
        b += 1
    return max(0, a * hop - int(HEAD * sr)), min(len(s), b * hop + int(TAIL * sr))


def gaps(s: np.ndarray, sr: int, min_s: float = 0.25):
    """Pauses inside the speech: stretches of >= min_s with no voice and nothing loud
    except perhaps a short consonant at either end.  [(start_s, length_s)]"""
    v, loud, _ = _layout(s, sr)
    out = []
    vi = np.flatnonzero(v)
    if len(vi) < 2:
        return out
    for a, b, on in _runs(v):
        if on or a <= vi[0] or b > vi[-1]:
            continue
        # the quiet middle of the gap (a consonant may sit at its edges)
        q = ~loud[a:b]
        if not q.any():
            continue
        qa, qb = a + int(np.argmax(q)), b - int(np.argmax(q[::-1]))
        if (qb - qa) * FRAME >= min_s:
            out.append((qa * FRAME, (qb - qa) * FRAME))
    return out


def talk_time(s: np.ndarray, sr: int) -> float:
    a, b = speech_span(s, sr)
    seg = s[a:b]
    return max(len(seg) / sr - HEAD - TAIL - sum(d for _, d in gaps(seg, sr)), 0.05)


def shorten_gaps(s: np.ndarray, sr: int, max_pause: float = INNER_MAX) -> np.ndarray:
    """Cut the middle out of pauses longer than max_pause (both edges are kept, so a
    consonant next to the pause is never touched)."""
    out, pos = [], 0
    keep = int(max_pause * sr / 2)
    for t0, d in gaps(s, sr, min_s=max_pause + 0.02):
        a, b = int(t0 * sr), int((t0 + d) * sr)
        out += [s[pos:a + keep]]
        f = min(int(0.01 * sr), keep)
        tail = s[b - keep:b].copy()
        if f:   # tiny cross-fade over the cut
            mix = np.linspace(0, 1, f, dtype=np.float32)
            head_end = out[-1][-f:].copy()
            out[-1] = out[-1][:-f]
            tail[:f] = head_end * (1 - mix) + tail[:f] * mix
        out.append(tail)
        pos = b
    out.append(s[pos:])
    return np.concatenate(out) if len(out) > 1 else s


def room_tone(s: np.ndarray, sr: int, n: int) -> np.ndarray:
    """n samples of the clip's own quietest sound, so a pause isn't dead digital silence
    in a clip that has a little hiss in it (a clean clip gets real silence)."""
    if n <= 0:
        return np.zeros(0, np.float32)
    db = _frame_db(s, sr)
    w = 12                                            # 120 ms
    if len(db) < w + 2:
        return np.zeros(n, np.float32)
    lvl = np.convolve(db, np.ones(w) / w, mode="valid")
    if lvl.min() < db.max() - 60:
        return np.zeros(n, np.float32)
    hop = int(sr * FRAME)
    i = int(np.argmin(lvl)) * hop
    piece = np.asarray(s[i:i + w * hop], np.float32)
    f = len(piece) // 4
    win = np.ones(len(piece), np.float32)
    win[:f] = np.linspace(0, 1, f)
    win[-f:] = np.linspace(1, 0, f)
    out = np.zeros(n + len(piece), np.float32)
    pos = 0
    while pos < n:
        out[pos:pos + len(piece)] += piece * win
        pos += len(piece) - f
    out = out[:n]
    g = min(int(0.03 * sr), n // 2)
    if g:
        out[:g] *= np.linspace(0, 1, g)
        out[-g:] *= np.linspace(1, 0, g)
    return out


# ── putting the pieces together ─────────────────────────────────────────────

def join(segs: list, texts: list, sr: int, rhythm: dict = None, slow=None):
    """Trim, even out and join the per-sentence pieces.  slow(audio, sr, stretch) slows a
    piece with the pitch kept (main passes intonation.match).  Returns (audio, note)."""
    r = rhythm or {}
    sent = float(np.clip(r.get("sentence", 0.45), *SENTENCE_PAUSE))
    comma = float(np.clip(r.get("comma", 0.22), *COMMA_PAUSE))
    out, speeds, slowed = [], [], 0
    for i, (raw, t) in enumerate(zip(segs, texts)):
        raw = np.asarray(raw, np.float32).reshape(-1)
        a, b = speech_span(raw, sr)
        s = shorten_gaps(raw[a:b], sr)
        rate = syllables(t) / talk_time(s, sr)
        if rate > MAX_RATE and slow is not None:
            stretch = min(rate / MAX_RATE, 1 / MIN_TEMPO)
            s2 = slow(s, sr, stretch)
            if s2 is not None and len(s2) > len(s):
                s = np.asarray(s2, np.float32)
                slowed += 1
        speeds.append(rate)
        if i:
            pause = comma if texts[i - 1].rstrip()[-1:] in ",;:" else sent
            # the untrimmed piece: its lead-in/tail hold the clip's plain room sound
            out.append(room_tone(raw, sr, int(sr * max(pause - HEAD - TAIL, 0.05))))
        f = min(int(0.005 * sr), len(s) // 2)
        if f:
            s = s.copy()
            s[:f] *= np.linspace(0, 1, f, dtype=np.float32)
            s[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
        out.append(s)
    note = (f"pace: {len(segs)} sentence(s) at {', '.join(f'{x:.1f}' for x in speeds)} syllables/s"
            + (f", {slowed} slowed to {MAX_RATE:g}" if slowed else "") + f"; pauses {sent:.2f}s")
    logger.info("Tamil %s", note)
    return np.concatenate(out) if out else np.zeros(0, np.float32), note
