from openserp import OpenSERP, SERPError
from smolagents import Tool, tool
import httpx
import time
from pathlib import Path


class OpenSERPWebPageSearchTool(Tool):
    name = "web_page_search"
    description = (
        "Search on web pages using OpenSERP and returns results list "
        "(title, link, snippet)."
    )
    inputs = {
        "query": {"type": "string", "description": "Search query."},
        "limit": {
            "type": "integer",
            "description": "How many results to return. (default: 5).",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, client: OpenSERP, **kwargs):
        super().__init__(**kwargs)
        self.client = client

    def forward(self, query: str, limit: int | None = None) -> str:
        try:
            resp = self.client.mega_search(engines=["google", "duckduckgo"], text=query, limit=limit or 5,
                                           mode="balanced")
        except SERPError as e:
            return f"Search error: {e}"

        if not resp.results:
            return "Nothing found."
        return "\n\n".join(
            f"{i}. [{r.title}]({r.url})\n{r.snippet or ''}"
            for i, r in enumerate(resp.results, 1)
        )


class OpenSERPImageSearchTool(Tool):
    name = "web_image_search"
    description = ("Search on web images using OpenSERP and returns a list of dicts with keys 'title', "
                   "'image_url' and 'thumbnail_url', e.g. results[0]['image_url']; an empty list if nothing "
                   "was found. The image_url is the full-size original on the source site and may fail to "
                   "download (403, hotlinking); the thumbnail_url is a small preview on the search engine CDN "
                   "that downloads reliably but may be None. Try the image_url first and fall back to the "
                   "thumbnail_url if the download fails.")
    inputs = {
        "query": {"type": "string", "description": "Search query."},
        "limit": {
            "type": "integer",
            "description": "How many results to return. (default: 5).",
            "nullable": True,
        },
    }
    # A list of dicts instead of formatted text: agents index results as dicts and waste steps parsing text
    output_type = "array"

    def __init__(self, client: OpenSERP, **kwargs):
        super().__init__(**kwargs)
        self.client = client

    def forward(self, query: str, limit: int | None = None) -> list[dict]:
        try:
            # Google image search hits captchas and DuckDuckGo times out on self-hosted OpenSERP
            resp = self.client.mega_image(engines=["bing", "yandex"], text=query, limit=limit or 5,
                                          mode="balanced")
        except SERPError as e:
            raise RuntimeError(f"Search error: {e}") from e

        return [
            {"title": r.title, "image_url": r.image.url, "thumbnail_url": r.image.thumbnail or None}
            for r in resp.results or []
            if r.image and r.image.url
        ]


# A good image downloads in 1-3 s; slow hosts used to hang for the full 30 s, so it's better to skip them
MAX_DOWNLOAD_TIMEOUT = 10.0
# A flashcard needs a few hundred KB; multi-MB originals (e.g. pexels' dl= links) trickle in for minutes
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024


@tool
def download_file_tool(url: str, filename: str, timeout: float = MAX_DOWNLOAD_TIMEOUT) -> str:
    """Download a file from a url and return it as a path string

    Args:
        url: The file url to download.
        filename: The path of the file to create after downloading, e.g. 'images/hund.jpg'.
        timeout: Total time limit in seconds for the whole download (default and maximum: 10).

    Returns:
        The file path of the downloaded file.
    """
    # Agents tend to pass timeout=30 by habit, so cap it
    timeout = min(timeout, MAX_DOWNLOAD_TIMEOUT)
    # httpx's timeout limits each network operation, not the whole download: a host that keeps sending
    # bytes slowly never trips it, so check the total time and size while streaming
    deadline = time.monotonic() + timeout
    chunks, size = [], 0
    try:
        # Many image hosts (e.g. Wikimedia) reject requests without a User-Agent
        with httpx.stream("GET", url, timeout=timeout, follow_redirects=True,
                          headers={"User-Agent": "AnkiAdder/0.1"}) as res:
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

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(chunks))
    return str(path)
