"""web_fetch: a web page as plain text, read in parts."""

import re
from html.parser import HTMLParser

from smolagents import Tool

from .net import fetch

PART_CHARS = 6000
MAX_PAGE_BYTES = 3 * 1024 * 1024
SKIP_TAGS = {"script", "style", "noscript", "svg", "template", "nav", "footer", "form", "button", "aside",
             "select", "iframe"}
# Their classes describe the whole page (Wikipedia: 'vector-feature-main-menu-pinned-disabled' on <html>)
CONTENT_TAGS = {"html", "body", "main", "article"}
# Menus, sidebars, language lists, cookie banners: the text of a page is elsewhere
SKIP_ROLES = {"navigation", "search", "menu", "menubar", "banner", "contentinfo", "dialog"}
SKIP_CLASS = re.compile(r"(^|[-_])(nav|navbar|navigation|menu|sidebar|footer|cookie|breadcrumbs?|toc|noprint|"
                        r"interlanguage|langlist|share|social)([-_]|$)")
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article", "header", "main",
              "blockquote", "pre", "dd", "dt", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "figcaption"}
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


def _skipped(tag: str, attrs: list[tuple[str, str | None]]) -> bool:
    if tag in SKIP_TAGS:
        return True
    if tag in CONTENT_TAGS:
        return False
    values = dict(attrs)
    if values.get("role") in SKIP_ROLES or values.get("aria-hidden") == "true" or "hidden" in values:
        return True
    names = f"{values.get('class') or ''} {values.get('id') or ''}".lower().split()
    return any(SKIP_CLASS.search(name) for name in names)


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._in_title = False
        # The element being skipped and how deep the same tag is nested inside it
        self._skip_tag: str | None = None
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if self._skip_tag:
            if tag == self._skip_tag:
                self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        elif tag not in VOID_TAGS and _skipped(tag, attrs):
            self._skip_tag, self._skip_depth = tag, 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")
            if tag in HEADINGS:
                self.parts.append("#" * int(tag[1]) + " ")
            elif tag == "li":
                self.parts.append("- ")

    def handle_endtag(self, tag):
        if self._skip_tag:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if self._skip_depth == 0:
                    self._skip_tag = None
            return
        if tag == "title":
            self._in_title = False
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip_tag:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    """The title and the readable text of an HTML page: one line per block, no empty lines or list items."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    lines = (re.sub(r"[ \t\r\f\v\xa0]+", " ", line).strip() for line in "".join(parser.parts).split("\n"))
    text = "\n".join(line for line in lines if line.strip("-# "))
    return parser.title.strip(), text


def decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


class WebFetchTool(Tool):
    name = "web_fetch"
    risk = "read"
    description = (f"Read a web page as plain text, {PART_CHARS} characters at a time, to check a fact on a page "
                   "you know the URL of (e.g. a Wikipedia article). Not for downloading media: use store_media.")
    inputs = {
        "url": {"type": "string", "description": "An http(s) URL."},
        "start": {"type": "integer", "nullable": True,
                  "description": "Character offset to read from, for the next part of a long page (default: 0)."},
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The last page read, so reading its next part costs no new download
        self._last: tuple[str, str, str] | None = None

    def forward(self, url: str, start: int | None = None) -> str:
        if not url.startswith(("http://", "https://")):
            raise ValueError("Pass an http(s) URL.")
        if self._last and self._last[0] == url:
            _, title, text = self._last
        else:
            data = fetch(url, max_bytes=MAX_PAGE_BYTES)
            if data.startswith(b"%PDF"):
                raise ValueError("This URL is a PDF file, not a web page: it cannot be read as text.")
            raw = decode(data)
            title, text = html_to_text(raw) if "<" in raw[:2000] else ("", raw.strip())
            self._last = (url, title, text)
        start = max(0, start or 0)
        part = text[start:start + PART_CHARS]
        end = start + len(part)
        header = f"Title: {title}\n" if title else ""
        header += f"Characters {start}-{end} of {len(text)}."
        if end < len(text):
            header += f" Call web_fetch again with start={end} for the next part."
        return header + "\n\n" + (part or "(no text)")
