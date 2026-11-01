"""The web tools of the assistant itself (image_searcher has its own)."""

from smolagents import Tool

from .dictionary import DictionaryLookupTool
from .net import WebCache
from .page import WebFetchTool


def web_tools(cache: WebCache) -> list[Tool]:
    return [DictionaryLookupTool(cache), WebFetchTool()]
