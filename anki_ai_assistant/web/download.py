import re
import time
from pathlib import Path

import httpx
from smolagents import Tool

# A good image downloads in 1-3 s; slow hosts used to hang for the full 30 s, so it's better to skip them
MAX_DOWNLOAD_TIMEOUT = 10.0
# A flashcard needs a few hundred KB; multi-MB originals (e.g. pexels' dl= links) trickle in for minutes
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024
USER_AGENT = "AnkiAIAssistant/0.1"


def safe_filename(filename: str) -> str:
    """Keep only the file name: the model may pass a path like 'images/hund.jpg' or '../x'."""
    name = re.sub(r"[^\w.\-]", "_", Path(filename.replace("\\", "/")).name).strip(".")
    if not name:
        raise ValueError(f"Invalid filename {filename!r}: pass a name like 'hund.jpg'.")
    return name


class DownloadFileTool(Tool):
    name = "download_file"
    description = "Download a file from a URL into the download folder and return the local path of the saved file."
    inputs = {
        "url": {"type": "string", "description": "The file URL to download."},
        "filename": {"type": "string", "description": "Name of the file to save, e.g. 'hund.jpg'."},
        "timeout": {
            "type": "number",
            "description": f"Total time limit in seconds for the whole download (default and maximum: {MAX_DOWNLOAD_TIMEOUT:g}).",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, download_dir: Path, **kwargs):
        super().__init__(**kwargs)
        self.download_dir = Path(download_dir)

    def forward(self, url: str, filename: str, timeout: float | None = None) -> str:
        path = self.download_dir / safe_filename(filename)
        # Agents tend to pass timeout=30 by habit, so cap it
        timeout = min(timeout or MAX_DOWNLOAD_TIMEOUT, MAX_DOWNLOAD_TIMEOUT)
        # httpx's timeout limits each network operation, not the whole download: a host that keeps sending
        # bytes slowly never trips it, so check the total time and size while streaming
        deadline = time.monotonic() + timeout
        chunks, size = [], 0
        try:
            # Many image hosts (e.g. Wikimedia) reject requests without a User-Agent
            with httpx.stream("GET", url, timeout=timeout, follow_redirects=True,
                              headers={"User-Agent": USER_AGENT}) as res:
                res.raise_for_status()
                for chunk in res.iter_bytes():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > MAX_DOWNLOAD_BYTES:
                        raise ValueError(f"File is larger than {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB: "
                                         "try the thumbnail_url or another result")
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Download took longer than {timeout:g} s: try another result")
        except httpx.TimeoutException as e:
            # httpx timeouts often have an empty message, which leaves the agent guessing
            raise TimeoutError(f"Download timed out after {timeout:g} s: try another result") from e

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"".join(chunks))
        return str(path)
