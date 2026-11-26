"""English pacing: a pause at every punctuation mark and a comfortable talking speed.

Measured problem: the English engine (Chatterbox-Turbo) copies a fast speaker's tempo
(4.7-5.4 syllables/s while talking, 190-225 words per minute) and is unreliable about
pauses: in one take a comma got 0.46 s, in the next 0.03 s; sentence ends got 0.61 s
in one take and 0.14 s in another.

Research basis:
  * Listeners prefer slowing speech with pauses at phrase breaks over stretching it
    evenly (Trouvain 2002), so pauses come first and tempo is only nudged.
  * Phrase breaks ~0.12-0.3 s, sentence/paragraph pauses ~0.4-0.7 s; ~0.6 s at
    sentence ends was rated most natural (PMC8874014).
  * Comfortable narration is ~150-160 words per minute, about 4.3-4.5 syllables per
    second while actually talking.

What is done (tested on the real model, two takes each):
  * one sentence per generation, joined with a fixed ~0.6 s pause (main._join_natural);
  * inside a sentence, a comma/semicolon/colon is shown to the engine as "..." -- it
    then makes its own breath (0.31 s and 0.39 s in both takes, vs 0.46/0.03 s for a
    plain comma).  The audio is never cut, and the user's text is unchanged;
  * if the talking is still faster than MAX_RATE, it is slowed (pitch kept) by at most
    MIN_TEMPO, in the same Praat resynthesis as the intonation matching.
"""
import re

SENTENCE_PAUSE = 0.45     # silence added between sentences; with each sentence's own breath at
                          # its edges (~0.18 s) the heard pause is ~0.6-0.7 s (slower speakers keep theirs)
COMMA_PAUSE = 0.30        # between pieces of an over-long sentence split at commas
MAX_RATE = 4.4            # syllables per second while talking (pauses excluded)
MIN_TEMPO = 0.85          # never slower than this
MIN_WORDS = 3             # shorter sentences are generated together with the next one


def _syll_word(w: str) -> int:
    w = re.sub(r"[^a-z]", "", w.lower())
    if not w:
        return 0
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(1, n)


def syllables(text: str) -> int:
    clean = re.sub(r"\[[^\]]*\]", " ", text)          # sound tags like [laugh] are not words
    return sum(_syll_word(w) for w in clean.split())


def _words(t: str) -> int:
    return len(re.sub(r"\[[^\]]*\]", " ", t).split())


def merge_short(pieces: list) -> list:
    """Glue a very short sentence ("Hi!") to the next one: a few-word input can make the
    engine ramble, and a greeting runs into the next sentence in real speech too."""
    out, carry = [], ""
    for p in pieces:
        cur = (carry + " " + p).strip() if carry else p
        if _words(cur) < MIN_WORDS:
            carry = cur
            continue
        out.append(cur)
        carry = ""
    if carry:
        if out:
            out[-1] = out[-1] + " " + carry
        else:
            out.append(carry)
    return out


def breath_marks(sentence: str) -> str:
    """What the engine reads: commas, semicolons and colons inside the sentence become
    '...', which the model reliably turns into a short natural pause."""
    return re.sub(r"(?<!\d)[,;:](?=\s+\S)", "...", sentence)


def talk_rate(s, sr: int, text: str):
    """Syllables per second while talking (pauses excluded), or None."""
    import numpy as np
    syl = syllables(text)
    if syl < 4:
        return None
    s = np.asarray(s, np.float32)
    hop = int(sr * 0.01)
    n = len(s) // hop
    if n < 20:
        return None
    db = 20 * np.log10(np.sqrt((s[: n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1)) + 1e-9)
    db = np.convolve(db, np.ones(3) / 3, mode="same")
    talking = np.sum(db > np.percentile(db, 90) - 30) * 0.01
    return syl / talking if talking > 0.5 else None


def tempo_for(rate, max_rate: float = MAX_RATE) -> float:
    """Tempo factor (<1 = slower) that brings the talking speed down to max_rate."""
    if not rate or rate <= max_rate:
        return 1.0
    return max(MIN_TEMPO, max_rate / rate)
