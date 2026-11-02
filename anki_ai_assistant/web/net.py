"""Network access shared by the web tools: a User-Agent, limits on time and size, a disk cache of API answers."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx

# A good image downloads in 1-3 s; slow hosts used to hang for the full 30 s, so it's better to skip them
MAX_DOWNLOAD_TIMEOUT = 10.0
# A flashcard needs a few hundred KB; multi-MB originals (e.g. pexels' dl= links) trickle in for minutes
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024
# Wikimedia rejects requests without a User-Agent and answers 429 to files it has not cached
# when the User-Agent has no contact URL
USER_AGENT = "AnkiAIAssistant/0.1 (Anki add-on; https://github.com/SolarianGG/AnkiAIAssistant)"
# Dictionary entries and search results hardly change within a week
CACHE_TTL = 7 * 24 * 3600
# A busy server (429 Too Many Requests, 503) is asked again this many times, after at most MAX_RETRY_DELAY s
RETRIES = 2
RETRY_STATUSES = (429, 503)
MAX_RETRY_DELAY = 5.0


class NotFound(Exception):
    """The server answered 404: the word or page does not exist there."""


def fetch(url: str, timeout: float | None = None, params: dict | None = None,
          max_bytes: int = MAX_DOWNLOAD_BYTES) -> bytes:
    """Download a file with a cap on the total time and size. A busy server (429, 503) is asked again
    after a short pause: Wikimedia answers 429 to a burst of downloads, e.g. images and then a recording."""
    for attempt in range(RETRIES + 1):
        try:
            return _fetch_once(url, timeout, params, max_bytes)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status not in RETRY_STATUSES:
                raise
            if attempt == RETRIES:
                raise RuntimeError(f"The server is busy ({status}) for {url}: try again later or another source") from e
            time.sleep(_retry_delay(e.response, attempt))
    raise AssertionError("unreachable")


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    try:
        delay = float(response.headers.get("Retry-After", ""))
    except ValueError:
        delay = 1.0 * 2 ** attempt
    return min(max(delay, 0.5), MAX_RETRY_DELAY)


def _fetch_once(url: str, timeout: float | None, params: dict | None, max_bytes: int) -> bytes:
    # Agents tend to pass timeout=30 by habit, so cap it
    timeout = min(timeout or MAX_DOWNLOAD_TIMEOUT, MAX_DOWNLOAD_TIMEOUT)
    # httpx's timeout limits each network operation, not the whole download: a host that keeps sending
    # bytes slowly never trips it, so check the total time and size while streaming
    deadline = time.monotonic() + timeout
    chunks, size = [], 0
    try:
        with httpx.stream("GET", url, params=params, timeout=timeout, follow_redirects=True,
                          headers={"User-Agent": USER_AGENT}) as res:
            if res.status_code == 404:
                raise NotFound(f"Not found (404): {res.url}")
            res.raise_for_status()
            for chunk in res.iter_bytes():
                chunks.append(chunk)
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError(f"File is larger than {max_bytes // (1024 * 1024)} MB: "
                                     "try the thumbnail_url or another result")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Download took longer than {timeout:g} s: try another result")
    except httpx.TimeoutException as e:
        # httpx timeouts often have an empty message, which leaves the agent guessing
        raise TimeoutError(f"Download timed out after {timeout:g} s: try another result") from e
    return b"".join(chunks)


class WebCache:
    """API answers on disk, so the same word looked up again costs no request."""

    def __init__(self, folder: Path | None, ttl: float = CACHE_TTL):
        self.folder = Path(folder) if folder else None
        self.ttl = ttl

    def _path(self, key: str) -> Path:
        return self.folder / (hashlib.sha256(key.encode("utf-8")).hexdigest()[:32] + ".json")

    def get(self, key: str) -> Any:
        if self.folder is None:
            return None
        path = self._path(key)
        try:
            if time.time() - path.stat().st_mtime > self.ttl:
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def put(self, key: str, value: Any) -> None:
        if self.folder is None:
            return
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            self._path(key).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass  # A cache that cannot be written only costs a request next time

    def get_text(self, url: str, params: dict | None = None, timeout: float | None = None) -> str:
        """GET a text answer, from the cache if it is fresh. A 404 is cached too and raises NotFound again."""
        key = url + "?" + json.dumps(params or {}, sort_keys=True, ensure_ascii=False)
        cached = self.get(key)
        if cached is not None:
            if cached.get("not_found"):
                raise NotFound(f"Not found (404): {url}")
            return cached["text"]
        try:
            text = fetch(url, timeout=timeout, params=params).decode("utf-8", errors="replace")
        except NotFound:
            self.put(key, {"not_found": True})
            raise
        self.put(key, {"text": text})
        return text

    def get_json(self, url: str, params: dict | None = None, timeout: float | None = None) -> Any:
        return json.loads(self.get_text(url, params, timeout))
