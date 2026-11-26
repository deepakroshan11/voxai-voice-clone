"""Activity log: every clone / voice-to-voice job is kept as a folder

    activity/<YYYY-MM-DD>/<HHMMSS>_<user>_<kind>/
        original.<ext>   the voice that was uploaded / recorded / picked (profile)
        cloned.wav       the result
        info.json        time, user id, language, text, voice source, lengths

locally under outputs/activity, and - when VOXAI_LOG_REPO is set (e.g. on the
Hugging Face Space, whose disk is wiped on restart) - uploaded in the background
to that PRIVATE Hugging Face dataset, where the owner can browse and play them.

"user" is an anonymous id: a salted hash of the visitor's IP address, so the same
visitor gets the same id without the IP itself being stored.
"""
import hashlib
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path

logger = logging.getLogger("voxai")

LOG_REPO = os.environ.get("VOXAI_LOG_REPO", "").strip()          # e.g. deepakroshan/voxai-activity
_SALT = os.environ.get("VOXAI_LOG_SALT", "voxai-activity")
_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "activity"
_lock = threading.Lock()


def user_id(request) -> str:
    """Anonymous, stable id for a visitor (hash of the client IP, never the IP)."""
    ip = ""
    try:
        fwd = request.headers.get("x-forwarded-for", "")
        ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "")
    except Exception:
        pass
    return "u-" + hashlib.sha256((_SALT + ip).encode()).hexdigest()[:8] if ip else "u-unknown"


def record(kind: str, user: str, original, original_name: str, result_path: str, info: dict):
    """Save one job; never raises (logging must not break cloning).
    original: bytes of the uploaded file, or a path to a file (e.g. a profile wav)."""
    try:
        now = time.gmtime()
        folder = _ROOT / time.strftime("%Y-%m-%d", now) / f"{time.strftime('%H%M%S', now)}_{user}_{kind}"
        folder.mkdir(parents=True, exist_ok=True)
        ext = (Path(original_name or "").suffix or ".wav").lower()[:6]
        if isinstance(original, (bytes, bytearray)):
            (folder / f"original{ext}").write_bytes(original)
        elif original and os.path.exists(original):
            shutil.copyfile(original, folder / f"original{Path(original).suffix or '.wav'}")
        if result_path and os.path.exists(result_path):
            shutil.copyfile(result_path, folder / "cloned.wav")
        meta = {"time_utc": time.strftime("%Y-%m-%d %H:%M:%S", now), "user": user, "kind": kind,
                "original_file": original_name or "", **info}
        (folder / "info.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Activity: %s by %s saved (%s)", kind, user, folder.name)
        if LOG_REPO:
            threading.Thread(target=_upload, args=(folder,), daemon=True).start()
    except Exception as e:
        logger.warning("Activity log failed: %s", e)


def _upload(folder: Path):
    """Copy one job folder to the private dataset (needs HF_TOKEN with write access)."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        with _lock:
            # Create the dataset only when missing: a token limited to writing this one
            # dataset may not be allowed to create repositories.
            if not api.repo_exists(LOG_REPO, repo_type="dataset"):
                api.create_repo(LOG_REPO, repo_type="dataset", private=True, exist_ok=True)
            rel = folder.relative_to(_ROOT).as_posix()
            api.upload_folder(repo_id=LOG_REPO, repo_type="dataset", folder_path=str(folder),
                              path_in_repo=f"activity/{rel}", commit_message=f"activity: {rel}")
        logger.info("Activity uploaded to %s: %s", LOG_REPO, rel)
    except Exception as e:
        logger.warning("Activity upload to %s failed: %s", LOG_REPO, e)
