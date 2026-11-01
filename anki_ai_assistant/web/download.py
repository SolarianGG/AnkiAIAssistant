import re
from pathlib import Path

from smolagents import Tool

from .net import MAX_DOWNLOAD_TIMEOUT, fetch


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
        data = fetch(url, timeout)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)
