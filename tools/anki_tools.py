import itertools
import json
import os
import time
from typing import Any

import httpx
from smolagents import Tool

DEFAULT_URL = "http://127.0.0.1:8765/api/v1"


class AnkiError(Exception):
    pass


class AnkiClient:
    """Minimal JSON-RPC 2.0 client for the Anki-Connect API v1."""

    def __init__(self, url: str | None = None, api_key: str | None = None, timeout: float = 60.0):
        self.url = url or os.getenv("ANKICONNECT_URL", DEFAULT_URL)
        api_key = api_key or os.getenv("ANKICONNECT_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(headers=headers, timeout=timeout)
        self._ids = itertools.count(1)

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        body = {"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": params or {}}
        try:
            response = self._http.post(self.url, json=body)
        except httpx.ConnectError as e:
            raise AnkiError(f"Cannot connect to Anki-Connect at {self.url}. Is Anki running?") from e
        response.raise_for_status()
        reply = response.json()
        if reply.get("error"):
            raise AnkiError(f"{method}: {reply['error'].get('message')}")
        return reply["result"]


def _item_error(item: dict) -> str:
    error = item["error"]
    issues = (error.get("data") or {}).get("issues")
    return f"{error.get('message')}" + (f" {json.dumps(issues, ensure_ascii=False)}" if issues else "")


class AnkiListDecksTool(Tool):
    name = "anki_list_decks"
    description = "List all Anki deck names, one per line. Subdecks are written as 'Parent::Child'."
    inputs = {}
    output_type = "string"

    def __init__(self, client: AnkiClient, **kwargs):
        super().__init__(**kwargs)
        self.client = client

    def forward(self) -> str:
        return "\n".join(deck["name"] for deck in self.client.call("decks.list"))


class AnkiListNoteTypesTool(Tool):
    name = "anki_list_note_types"
    description = (
        "List Anki note types with their field names, e.g. 'Basic: Front, Back'. "
        "Use it to know which field names anki_add_notes expects."
    )
    inputs = {}
    output_type = "string"

    def __init__(self, client: AnkiClient, **kwargs):
        super().__init__(**kwargs)
        self.client = client

    def forward(self) -> str:
        lines = []
        for note_type in self.client.call("noteTypes.list"):
            info = self.client.call("noteTypes.get", {"noteType": {"id": note_type["id"]}})
            lines.append(f"{info['name']}: {', '.join(f['name'] for f in info['fields'])}")
        return "\n".join(lines)


class AnkiAddNotesTool(Tool):
    name = "anki_add_notes"
    description = (
        "Add notes (cards) to an Anki deck. The deck is created if it does not exist. "
        "Each note is a dict mapping field names of the note type to values, "
        "e.g. [{'Front': 'Haus', 'Back': 'house'}] for the 'Basic' note type. "
        "Notes are added independently; returns one line per note with its id or the error."
    )
    inputs = {
        "deck": {"type": "string", "description": "Deck name, e.g. 'German' or 'Languages::German'."},
        "notes": {
            "type": "array",
            "description": "List of notes, each a dict {field name: value}. Values may contain HTML.",
        },
        "note_type": {
            "type": "string",
            "description": "Note type name (default: 'Basic'). See anki_list_note_types.",
            "nullable": True,
        },
        "tags": {"type": "array", "description": "Tags for all notes, e.g. ['german'].", "nullable": True},
    }
    output_type = "string"

    def __init__(self, client: AnkiClient, **kwargs):
        super().__init__(**kwargs)
        self.client = client

    def forward(self, deck: str, notes: list, note_type: str | None = None, tags: list | None = None) -> str:
        note_type = note_type or "Basic"
        # Fail on an unknown note type before creating the deck
        self.client.call("noteTypes.get", {"noteType": {"name": note_type}})
        # decks.create returns the existing deck for an exact existing name
        deck_id = self.client.call("decks.create", {"name": deck})["id"]
        payload = [
            {
                "noteType": {"name": note_type},
                "deck": {"id": deck_id},
                "fields": {k: str(v) for k, v in note.items()},
                "tags": tags or [],
            }
            for note in notes
        ]
        items = self.client.call("notes.createMany", {"notes": payload})["items"]

        lines = []
        for note, item in zip(notes, items):
            label = json.dumps(note, ensure_ascii=False)
            if "error" in item:
                lines.append(f"FAILED {label}: {_item_error(item)}")
            else:
                lines.append(f"OK {label}: noteId={item['result']['noteId']}")
        if all("error" in item for item in items):
            raise AnkiError("No notes were added:\n" + "\n".join(lines))
        return "\n".join(lines)


class AnkiSearchNotesTool(Tool):
    name = "anki_search_notes"
    description = (
        "Search Anki notes with Anki search syntax and return their fields, e.g. "
        "'deck:\"German\"', 'Front:Haus', 'tag:german', 'deck:German Haus'. "
        "Useful to check for existing cards before adding new ones."
    )
    inputs = {
        "query": {"type": "string", "description": "Anki search query."},
        "limit": {"type": "integer", "description": "Max notes to return (default: 20).", "nullable": True},
    }
    output_type = "string"

    def __init__(self, client: AnkiClient, **kwargs):
        super().__init__(**kwargs)
        self.client = client

    def forward(self, query: str, limit: int | None = None) -> str:
        ids = self.client.call("notes.search", {"query": query, "limit": limit or 20})["ids"]
        if not ids:
            return "No notes found."
        items = self.client.call("notes.getMany", {"ids": ids})["items"]
        return "\n".join(
            json.dumps({"id": i["result"]["id"], "noteType": i["result"]["noteType"]["name"],
                        "fields": i["result"]["fields"]}, ensure_ascii=False)
            for i in items if "result" in i
        )


class AnkiStoreMediaTool(Tool):
    name = "anki_store_media"
    description = (
        "Store an image (or other media file) in Anki's media collection from a URL or a local file path. "
        "Returns the stored filename; put it into a note field as <img src=\"FILENAME\">. "
        "If storing from a URL fails, download the file first and pass its local path."
    )
    inputs = {
        "filename": {"type": "string", "description": "Desired filename, e.g. 'perro.jpg'. May be renamed on collision."},
        "source": {"type": "string", "description": "http(s) URL or local file path of the file."},
    }
    output_type = "string"

    def __init__(self, client: AnkiClient, **kwargs):
        super().__init__(**kwargs)
        self.client = client

    def forward(self, filename: str, source: str) -> str:
        src = {"url": source} if source.startswith(("http://", "https://")) else {"path": os.path.abspath(source)}
        # media.store is asynchronous: it returns an operation to poll
        op_id = self.client.call("media.store", {"filename": filename, "source": src})["operationId"]
        while True:
            op = self.client.call("operations.get", {"operationId": op_id})
            if op["state"] == "succeeded":
                return op["result"]["filename"]
            if op["state"] in ("failed", "cancelled"):
                raise AnkiError(f"media.store {op['state']}: {op['error'].get('message')}")
            time.sleep(0.2)


def anki_tools(client: AnkiClient | None = None) -> list[Tool]:
    client = client or AnkiClient()
    return [
        AnkiListDecksTool(client),
        AnkiListNoteTypesTool(client),
        AnkiAddNotesTool(client),
        AnkiSearchNotesTool(client),
        AnkiStoreMediaTool(client),
    ]
