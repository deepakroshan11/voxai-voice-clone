"""Automatic transcript of the voice sample.

IndicF5 (F5-TTS) continues speech *after* the reference clip, aligned against
"reference transcript + new text".  If the transcript does not match what is
actually said in the clip, the alignment shifts and words of the new text get
swallowed or garbled.  So when the user gives no transcript we produce one:

  1. openai/whisper-small detects the spoken language (and transcribes English).
  2. Tamil speech is transcribed by vasista22/whisper-tamil-small, a Whisper
     fine-tuned on Tamil that writes what was *said* (spoken forms like
     "பேசுறது"), not a formalised paraphrase -- exactly what F5 needs.

Models are loaded on demand and released afterwards (the laptop has 8 GB RAM);
results are cached per voice so a profile is only transcribed once.
"""
import gc
import logging
import threading

logger = logging.getLogger("voxai")

LID_REPO = "openai/whisper-small"
TA_REPO = "vasista22/whisper-tamil-small"
_lock = threading.Lock()
_cache = {}


def cer(ref: str, hyp: str) -> float:
    """Character error rate over letters + vowel signs (punctuation/spaces ignored)."""
    import unicodedata
    r = [c for c in ref.lower() if unicodedata.category(c)[0] in "LM"]
    h = [c for c in hyp.lower() if unicodedata.category(c)[0] in "LM"]
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))


def _load(repo):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig
    proc = WhisperProcessor.from_pretrained(repo)
    model = WhisperForConditionalGeneration.from_pretrained(repo).eval()
    if repo != LID_REPO:   # the fine-tune ships an outdated generation config
        model.generation_config = GenerationConfig.from_pretrained(LID_REPO)
    return proc, model


def transcribe(path: str, key: str = None) -> tuple:
    """Return (language_code, transcript) for the speech in `path`."""
    if key and key in _cache:
        return _cache[key]
    import librosa
    import torch
    with _lock:
        audio, _ = librosa.load(path, sr=16000)
        audio = audio[: 30 * 16000]
        proc, model = _load(LID_REPO)
        feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features
        with torch.inference_mode():
            lang_ids = model.detect_language(feats)
            lang = proc.tokenizer.convert_ids_to_tokens(int(lang_ids[0])).strip("<|>")
            if lang != "ta":
                ids = model.generate(feats, language=lang, task="transcribe", max_new_tokens=220)
                text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        del model
        gc.collect()
        if lang == "ta":
            proc, model = _load(TA_REPO)
            feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features
            with torch.inference_mode():
                ids = model.generate(feats, language="ta", task="transcribe", max_new_tokens=220)
            text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
            del model
            gc.collect()
    logger.info("Transcribed voice sample [%s]: %s", lang, text)
    if key:
        _cache[key] = (lang, text)
    return lang, text
