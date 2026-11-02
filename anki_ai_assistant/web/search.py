"""Web and image search: Wikipedia and Wikimedia Commons, and a self-hosted OpenSERP server if there is one."""

from pathlib import PurePosixPath
from typing import Any

from smolagents import Tool

from ..core.logs import log, short
from .net import WebCache

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
# Big enough for a card; store_media shrinks it further
IMAGE_WIDTH = 800


def _clean(url: str) -> str:
    """Drop Wikimedia's tracking parameters (?utm_source=...): they would end up in the stored file name."""
    return url.split("?", 1)[0]


def wikipedia_images(cache: WebCache, query: str, limit: int) -> list[dict]:
    """The lead images of the Wikipedia articles found for the query: for a common noun (dog, apple) the article
    shows one typical example."""
    answer = cache.get_json(WIKIPEDIA_API, params={
        "action": "query", "format": "json", "redirects": 1, "generator": "search", "gsrsearch": query,
        "gsrlimit": limit, "prop": "pageimages", "piprop": "thumbnail|original", "pithumbsize": IMAGE_WIDTH})
    pages = sorted((answer.get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    results = []
    for page in pages:
        original = (page.get("original") or {}).get("source", "")
        thumbnail = (page.get("thumbnail") or {}).get("source")
        # SVG lead images are mostly logos, maps and flags
        if not thumbnail or _clean(original).lower().endswith(".svg"):
            continue
        results.append({"title": f"Wikipedia article '{page['title']}'", "image_url": _clean(thumbnail),
                        "thumbnail_url": None})
    return results


def commons_images(cache: WebCache, query: str, limit: int) -> list[dict]:
    answer = cache.get_json(COMMONS_API, params={
        "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
        "gsrsearch": f"{query} filetype:bitmap", "gsrlimit": limit, "prop": "imageinfo", "iiprop": "url",
        "iiurlwidth": IMAGE_WIDTH})
    pages = sorted((answer.get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    results = []
    for page in pages:
        info = (page.get("imageinfo") or [{}])[0]
        url = info.get("thumburl") or info.get("url")
        if url:
            title = PurePosixPath(page.get("title", "").removeprefix("File:")).stem
            results.append({"title": title, "image_url": _clean(url), "thumbnail_url": None})
    return results


def wikimedia_images(cache: WebCache, query: str, limit: int) -> list[dict]:
    return wikipedia_images(cache, query, limit) + commons_images(cache, query, limit)


class WebImageSearchTool(Tool):
    name = "web_image_search"
    description = ("Search images on the web. Returns a list of dicts with keys 'title', 'image_url' and "
                   "'thumbnail_url', e.g. results[0]['image_url']; an empty list if nothing was found. "
                   "The image_url may fail to download (403, hotlinking); the thumbnail_url is a small preview "
                   "that downloads reliably but may be None. Try the image_url first and fall back to the "
                   "thumbnail_url if the download fails.")
    inputs = {
        "query": {"type": "string", "description": "What to search for, in English, e.g. 'dog'."},
        "limit": {"type": "integer", "description": "How many results to return per source (default: 5).",
                  "nullable": True},
    }
    # A list of dicts instead of formatted text: agents index results as dicts and waste steps parsing text
    output_type = "array"

    def __init__(self, cache: WebCache, openserp: Any = None, **kwargs):
        super().__init__(**kwargs)
        self.cache = cache
        self.openserp = openserp

    def forward(self, query: str, limit: int | None = None) -> list[dict]:
        limit = max(1, min(limit or 5, 10))
        if self.openserp is not None:
            try:
                # Google image search hits captchas and DuckDuckGo times out on self-hosted OpenSERP
                resp = self.openserp.mega_image(engines=["bing", "yandex"], text=query, limit=limit, mode="balanced")
                results = [{"title": r.title, "image_url": r.image.url, "thumbnail_url": r.image.thumbnail or None}
                           for r in resp.results or [] if r.image and r.image.url]
                if results:
                    return results
                log.info("OpenSERP found no images for %r: searching Wikimedia", query)
            except Exception as e:
                # OpenSERP is often not running: Wikimedia needs no server of its own.
                # The model sees only the Wikimedia results, so the failure is visible in the log alone
                log.warning("OpenSERP image search failed for %r (%s: %s): searching Wikimedia",
                            query, type(e).__name__, short(e, 500))
        return wikimedia_images(self.cache, query, limit)


class OpenSERPWebPageSearchTool(Tool):
    name = "web_page_search"
    description = "Search web pages using OpenSERP; returns a list of results (title, link, snippet)."
    inputs = {
        "query": {"type": "string", "description": "Search query."},
        "limit": {"type": "integer", "description": "How many results to return (default: 5).", "nullable": True},
    }
    output_type = "string"

    def __init__(self, client: Any, **kwargs):
        super().__init__(**kwargs)
        self.client = client

    def forward(self, query: str, limit: int | None = None) -> str:
        from openserp import SERPError

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

