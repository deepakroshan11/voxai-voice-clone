"""Intonation matching: give a flat clone back the speaker's own melody range.

Measured problem: Tamil clones of male voices moved their pitch ~1.1-1.3 semitones (SD)
while the same speakers really move ~3.5-5.5 -- the words are right but the voice sounds
read by a machine.  English clones were already within the speaker's range.

Method (classic prosody modification, as done in Praat):
  1. Pitch is measured with a two-pass range (De Looze & Hirst): pass 1 finds the
     speaker's median, pass 2 tracks with floor/ceiling around it, so male and female
     voices are both followed without octave jumps.
  2. The speaker's own intonation range is measured on the voice sample.
  3. If the clone is flatter, only the slow phrase melody (rises, falls, stress peaks:
     the log-F0 contour smoothed over ~120 ms) is widened around the median; the fast
     micro-movements that make a voice sound human are kept as they are.
  4. The new contour is applied with PSOLA (overlap-add), which keeps the voice's
     timbre and timing.  Lively clones are left untouched.
"""
import logging

import numpy as np

logger = logging.getLogger("voxai")

MAX_GAIN = 1.9          # never stretch the melody more than this
TARGET_SHARE = 0.85     # aim at 85% of the speaker's range (a reading is a bit calmer than talk)
TARGET_CAP_ST = 4.2     # very animated samples (film clips) are not copied beyond this
MIN_VOICED = 30         # pitch frames needed before anything is measured or changed


def _sound(y: np.ndarray, sr: int):
    import parselmouth
    return parselmouth.Sound(np.asarray(y, np.float64), sampling_frequency=sr)


def _track(snd):
    """Two-pass pitch track -> (times, Hz) of voiced frames, plus (floor, ceiling)."""
    p1 = snd.to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=600)
    f = p1.selected_array["frequency"]
    f = f[f > 0]
    if len(f) < MIN_VOICED:
        return None, None, (60, 600)
    q15, q85 = np.percentile(f, [15, 85])
    floor, ceil = max(50.0, 0.75 * q15), min(700.0, 1.6 * q85)
    p2 = snd.to_pitch_ac(time_step=0.01, pitch_floor=floor, pitch_ceiling=ceil)
    hz = p2.selected_array["frequency"]
    t = p2.xs()
    v = hz > 0
    return t[v], hz[v], (floor, ceil)


def _runs(t, gap=0.035):
    """Indices splitting the voiced frames into continuous runs."""
    br = np.where(np.diff(t) > gap)[0] + 1
    return np.split(np.arange(len(t)), br)


def _smooth(t, st, sigma_s=0.06):
    """Slow phrase melody: log-F0 smoothed inside each voiced run (Gaussian, ~120 ms wide)."""
    out = st.copy()
    for idx in _runs(t):
        if len(idx) < 3:
            continue
        x = t[idx]
        w = np.exp(-0.5 * ((x[:, None] - x[None, :]) / sigma_s) ** 2)
        out[idx] = (w @ st[idx]) / w.sum(1)
    return out


def melody_track(y: np.ndarray, sr: int):
    """(times, slow melody in semitones) of the whole clip, or None."""
    try:
        t, hz, _ = _track(_sound(y, sr))
    except Exception as e:
        logger.warning("Intonation track: %s", e)
        return None
    if t is None:
        return None
    return t, _smooth(t, 12 * np.log2(hz / np.median(hz)))


def median_hz(y: np.ndarray, sr: int):
    """The speaker's pitch level: median F0 of the voiced speech (Hz), or None."""
    try:
        t, hz, _ = _track(_sound(y, sr))
    except Exception as e:
        logger.warning("Pitch level: %s", e)
        return None
    return float(np.median(hz)) if t is not None else None


def melody_sd(y: np.ndarray, sr: int):
    """SD (semitones) of the phrase melody, or None if there is too little voiced speech."""
    try:
        t, hz, _ = _track(_sound(y, sr))
    except Exception as e:
        logger.warning("Intonation measure: %s", e)
        return None
    if t is None:
        return None
    st = 12 * np.log2(hz / np.median(hz))
    return float(np.std(_smooth(t, st)))


def _resynth(y, snd, t, st, slow, med, floor, ceil, k, stretch=1.0):
    """PSOLA resynthesis, in one pass: melody widened k times (micro-movement unchanged)
    and duration multiplied by `stretch` (pitch kept).
    (Moving the clone's pitch LEVEL to the speaker's was tested and dropped: it lowered
    speaker similarity, e.g. 0.751 -> 0.725 -- the level comes with matching formants.)"""
    from parselmouth.praat import call
    manip = call(snd, "To Manipulation", 0.01, floor, ceil)
    if k != 1.0 and t is not None:
        new_hz = np.clip(med * 2 ** ((k * slow + (st - slow)) / 12), floor * 0.8, ceil * 1.25)
        tier = call("Create PitchTier", "melody", snd.xmin, snd.xmax)
        for ti, fi in zip(t, new_hz):
            call(tier, "Add point", float(ti), float(fi))
        call([tier, manip], "Replace pitch tier")
    if abs(stretch - 1.0) > 1e-3:
        dt = call("Create DurationTier", "tempo", snd.xmin, snd.xmax)
        call(dt, "Add point", float(snd.xmin), float(stretch))
        call([dt, manip], "Replace duration tier")
    out = call(manip, "Get resynthesis (overlap-add)").values[0].astype(np.float32)
    n = int(round(len(y) * stretch))
    out = np.pad(out, (0, max(0, n - len(out))))[:n]
    return out * min(1.0, (np.abs(y).max() + 1e-9) / (np.abs(out).max() + 1e-9))   # same peak


def match(y: np.ndarray, sr: int, speaker_sd, stretch: float = 1.0):
    """One PSOLA pass that widens a flat clone's melody toward the speaker's range and,
    if stretch > 1, slows it down (pitch kept).  Returns (audio, note)."""
    slower = abs(stretch - 1.0) > 0.02
    pace_note = f"; slowed x{1 / stretch:.2f}" if slower else ""
    try:
        snd = _sound(y, sr)
        t, hz, (floor, ceil) = _track(snd)
        k, note = 1.0, "intonation: no reference measure"
        med = st = slow = None
        if t is None:
            note = "intonation: too little voiced speech"
        else:
            med = float(np.median(hz))
            st = 12 * np.log2(hz / med)
            slow = _smooth(t, st)
            if speaker_sd:
                clone_sd = float(np.std(slow))
                target = min(speaker_sd * TARGET_SHARE, TARGET_CAP_ST)
                if clone_sd <= 0.05 or clone_sd >= target * 0.95:
                    note = f"intonation: kept ({clone_sd:.1f} st vs speaker {speaker_sd:.1f})"
                else:
                    k = float(min(target / clone_sd, MAX_GAIN))
        if k == 1.0 and not slower:
            return y, note
        out = _resynth(y, snd, t, st, slow, med, floor, ceil, k, stretch if slower else 1.0)
        if k == 1.0:
            logger.info("Pace: %s%s", note, pace_note)
            return out, note + pace_note
        got = melody_sd(out, sr) or clone_sd
        want = clone_sd * k
        # PSOLA delivers only part of a requested change: measure it once and redo the
        # resynthesis from the original with the corrected amount (one pass in the output).
        if got < want * 0.92 and got > clone_sd + 0.05:
            eff = (got - clone_sd) / (want - clone_sd)
            k2 = float(min(1 + (k - 1) / eff, MAX_GAIN * 1.3))
            out2 = _resynth(y, snd, t, st, slow, med, floor, ceil, k2, stretch if slower else 1.0)
            got2 = melody_sd(out2, sr) or got
            if got2 > got:
                out, got, k = out2, got2, k2
        logger.info("Intonation: melody %.1f -> %.1f st (x%.2f, speaker %.1f st)%s", clone_sd, got, k, speaker_sd, pace_note)
        return out, f"intonation: {clone_sd:.1f} -> {got:.1f} st (speaker {speaker_sd:.1f}){pace_note}"
    except Exception as e:
        logger.warning("Intonation/pace step skipped: %s", e)
        return y, "intonation: skipped"
