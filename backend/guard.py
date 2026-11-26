"""Protections for the public deployment.

* Upload cap and real-audio check: only known audio containers reach ffmpeg, with its
  format forced and network/playlist inputs disabled, so a crafted "audio" file cannot
  make ffmpeg read other files on the server.
* Private voice profiles: each browser sends a random id (X-Client-Id); its profiles
  live in their own folder, so visitors never see, play, use or delete each other's voices.
* One heavy job at a time, with a visible place in line, a cap on how many can wait,
  and a per-visitor limit on clones.

Defaults are strict on Hugging Face Spaces (SPACE_ID is set there) and relaxed for a
local single-user run; every switch can be overridden with an environment variable.
"""
import collections
import contextlib
import hashlib
import os
import re
import threading
import time
import uuid

from fastapi import HTTPException

ON_SPACE = bool(os.environ.get("SPACE_ID"))


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


MAX_UPLOAD_MB = float(os.environ.get("VOXAI_MAX_UPLOAD_MB", "40"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
PRIVATE_PROFILES = _flag("VOXAI_PRIVATE_PROFILES", ON_SPACE)
REQUIRE_CONSENT = _flag("VOXAI_REQUIRE_CONSENT", ON_SPACE)
MAX_PROFILES = int(os.environ.get("VOXAI_MAX_PROFILES", "12"))     # per browser


def allowed_origins() -> list:
    env = os.environ.get("VOXAI_ORIGINS")
    if env:
        return [o.strip() for o in env.split(",") if o.strip()]
    out = [f"http://{h}:{p}" for h in ("localhost", "127.0.0.1") for p in (8000, 7860)]
    if os.environ.get("SPACE_HOST"):
        out.append(f"https://{os.environ['SPACE_HOST']}")
    return out


def origin_allowed(origin: str, headers) -> bool:
    """Browser requests from other websites are refused by the app itself: on Spaces the
    Hugging Face proxy answers CORS preflights for every origin, so the CORS headers alone
    do not stop them.  Requests without an Origin (curl, same-origin GETs) pass; the rate
    limit and queue cover those."""
    if not origin:
        return True
    if origin in allowed_origins():
        return True
    host = origin.split("://", 1)[-1].rstrip("/").lower()
    same = {h.strip().lower() for h in (headers.get("x-forwarded-host", ""), headers.get("host", "")) if h}
    return host in same


# ---------------------------------------------------------------- uploads
def read_upload(upload) -> bytes:
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_MB:g} MB) -- use a shorter clip")
    if not data:
        raise HTTPException(400, "The audio file is empty")
    return data


def audio_format(data: bytes):
    """ffmpeg demuxer for a real audio container, judged by its first bytes; None otherwise."""
    h = data[:16]
    if h[:4] == b"RIFF" and h[8:12] == b"WAVE":
        return "wav"
    if h[:4] == b"OggS":
        return "ogg"
    if h[:4] == b"fLaC":
        return "flac"
    if h[:4] == b"\x1aE\xdf\xa3":
        return "matroska"                       # also WebM (browser recordings)
    if h[4:8] == b"ftyp":
        return "mov"                            # MP4 / M4A (external references stay off)
    if h[:3] == b"ID3":
        return "mp3"
    if h[:8] == b"\x30\x26\xb2\x75\x8e\x66\xcf\x11":
        return "asf"                            # WMA
    if h[:5] == b"#!AMR":
        return "amr"
    if len(h) > 1 and h[0] == 0xFF and (h[1] & 0xE0) == 0xE0:
        return "aac" if (h[1] & 0x06) == 0 else "mp3"   # ADTS AAC vs MPEG audio frame
    return None


def ffmpeg_input_args(fmt: str) -> list:
    """Input options for an uploaded file: fixed demuxer, local file only, no stdin."""
    return ["-nostdin", "-protocol_whitelist", "file", "-f", fmt]


# ---------------------------------------------------------------- private profiles
_CID_RE = re.compile(r"^[A-Za-z0-9-]{16,64}$")
_OWNER_SALT = os.environ.get("VOXAI_LOG_SALT", "voxai-activity") + "|profiles"


def owner_id(request) -> str:
    cid = (request.headers.get("x-client-id") or "").strip()
    if not _CID_RE.match(cid):
        raise HTTPException(400, "Missing browser id -- please reload the page")
    return "c-" + hashlib.sha256((_OWNER_SALT + cid).encode()).hexdigest()[:20]


# ---------------------------------------------------------------- consent
def check_consent(value) -> bool:
    ok = str(value or "").strip().lower() in ("1", "true", "yes", "on")
    if REQUIRE_CONSENT and not ok:
        raise HTTPException(400, "Please confirm you have permission to use this voice")
    return ok


# ---------------------------------------------------------------- rate limit
class RateLimit:
    """At most n jobs per window seconds for one visitor ("0/..." switches it off)."""

    def __init__(self, spec: str):
        n, w = (spec or "0/1").split("/")
        self.n, self.w = int(n), float(w)
        self.hits = collections.defaultdict(collections.deque)
        self.lock = threading.Lock()

    def check(self, key: str):
        if self.n <= 0:
            return
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > self.w:
                q.popleft()
            if len(q) >= self.n:
                wait_min = max(1, round((self.w - (now - q[0])) / 60))
                raise HTTPException(429, f"Limit reached ({self.n} clones per {self.w / 60:g} min) -- "
                                         f"try again in about {wait_min} min")
            q.append(now)
            if len(self.hits) > 5000:                 # forget idle visitors
                for k in [k for k, v in self.hits.items() if not v or now - v[-1] > self.w]:
                    del self.hits[k]


RATE = RateLimit(os.environ.get("VOXAI_RATE_LIMIT", "5/600" if ON_SPACE else "0/1"))


# ---------------------------------------------------------------- job queue
_TICKET_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")


class JobQueue:
    """Heavy work runs `slots` at a time (FIFO); others wait and can ask their place in line."""

    def __init__(self, slots: int, max_waiting: int):
        self.cond = threading.Condition()
        self.slots, self.max_waiting = slots, max_waiting
        self.waiting = collections.OrderedDict()
        self.running = set()

    @contextlib.contextmanager
    def slot(self, ticket: str = None):
        ticket = ticket if ticket and _TICKET_RE.match(ticket) else uuid.uuid4().hex
        with self.cond:
            if ticket in self.waiting or ticket in self.running:
                ticket = uuid.uuid4().hex
            if len(self.waiting) >= self.max_waiting:
                raise HTTPException(503, "VOXAI is busy right now -- please try again in a few minutes")
            self.waiting[ticket] = time.monotonic()
            while len(self.running) >= self.slots or next(iter(self.waiting)) != ticket:
                self.cond.wait()
            del self.waiting[ticket]
            self.running.add(ticket)
        try:
            yield
        finally:
            with self.cond:
                self.running.discard(ticket)
                self.cond.notify_all()

    def status(self, ticket: str) -> dict:
        with self.cond:
            if ticket in self.running:
                return {"state": "running", "ahead": 0}
            if ticket in self.waiting:
                return {"state": "waiting", "ahead": len(self.running) + list(self.waiting).index(ticket)}
            return {"state": "unknown", "ahead": 0, "busy": len(self.running) + len(self.waiting)}


JOBS = JobQueue(int(os.environ.get("VOXAI_MAX_JOBS", "1")), int(os.environ.get("VOXAI_MAX_WAITING", "8")))
