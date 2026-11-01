"""Web and image search through a self-hosted OpenSERP server."""

from openserp import OpenSERP, SERPError
from smolagents import Tool


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
