"""Web tools without the network: the services' answers are faked."""

import json
import os
import time
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image
from smolagents.models import get_tool_json_schema

from anki_ai_assistant.web import net, page
from anki_ai_assistant.web.dictionary import (COMMONS_API, DictionaryLookupTool, compact_entry, kaikki_url,
                                              resolve_audio, resolve_language)
from anki_ai_assistant.web.media import prepare_image
from anki_ai_assistant.web.net import NotFound, WebCache
from anki_ai_assistant.web.page import WebFetchTool, html_to_text
from anki_ai_assistant.web.search import WebImageSearchTool
from anki_ai_assistant.web.tools import web_tools

HUND = {
    "word": "Hund", "pos": "noun", "lang": "German",
    "head_templates": [{"name": "de-noun", "expansion": "Hund m (strong, genitive Hundes, plural Hunde)"}],
    "sounds": [{"ipa": "/hʊnt/"}, {"rhymes": "-ʊnt"},
               {"audio": "De-Hund.ogg", "ogg_url": "https://x/De-Hund.ogg", "mp3_url": "https://x/De-Hund.ogg.mp3"},
               {"audio": "De-at-Hund.ogg", "tags": ["Austria"], "ogg_url": "https://x/De-at-Hund.ogg"}],
    "forms": [{"form": "Hundes", "tags": ["genitive"]}, {"form": "Hunde", "tags": ["plural"]},
              {"form": "Hünde", "tags": ["error-unknown-tag", "plural"]},
              {"form": "de-ndecl", "source": "declension", "tags": ["table-tags"]},
              {"form": "Hunde", "tags": ["plural"]}],
    "senses": [{"glosses": ["dog, hound"], "tags": ["masculine", "strong"]},
               {"glosses": ["scoundrel"], "tags": ["derogatory", "masculine", "strong"],
                "examples": [{"text": "ein gemeiner Hund", "translation": "a mean dog"}]}],
}

TATOEBA = {"data": [
    {"text": "Ich will Hunde.", "translations": [{"lang": "fra", "text": "Je veux des chiens."},
                                                 {"lang": "eng", "text": "I want dogs.", "is_direct": True}]},
    {"text": "Unapproved Hund.", "is_unapproved": True, "translations": [{"lang": "eng", "text": "x"}]},
    {"text": "Hunde rennen schnell.", "translations": [{"lang": "rus", "text": "Собаки бегают быстро."}]},
]}


class FakeCache(WebCache):
    """Answers by URL; anything else is a 404."""

    def __init__(self, answers: dict):
        super().__init__(None)
        self.answers = answers
        self.requests = []

    def get_text(self, url, params=None, timeout=None):
        self.requests.append((url, params))
        if url not in self.answers:
            raise NotFound(url)
        answer = self.answers[url]
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, str) else json.dumps(answer)


def test_web_tools_are_valid_smolagents_tools():
    for tool in web_tools(WebCache(None)):
        assert tool.risk == "read"
        assert get_tool_json_schema(tool)["function"]["name"] == tool.name


def test_resolve_language():
    assert resolve_language("de") == ("German", "deu")
    assert resolve_language("german") == ("German", "deu")
    assert resolve_language("rus") == ("Russian", "rus")
    assert resolve_language("Norwegian") == ("Norwegian Bokmål", "nob")
    assert resolve_language("old english") == ("Old english", None)


def test_kaikki_url():
    assert kaikki_url("Hund", "German") == "https://kaikki.org/dictionary/German/meaning/H/Hu/Hund.jsonl"
    assert kaikki_url("a", "English").endswith("/English/meaning/a/a/a.jsonl")
    assert "/%D1%81/%D1%81%D0%BE/" in kaikki_url("собака", "Russian")


def test_compact_entry():
    entry = compact_entry(HUND)
    assert entry["head"] == "Hund m (strong, genitive Hundes, plural Hunde)"
    assert entry["ipa"] == ["/hʊnt/"]
    # No table headers, no duplicates; the 'error' tag is dropped
    assert entry["forms"] == ["Hundes (genitive)", "Hunde (plural)", "Hünde (plural)"]
    # Tags every sense has are grammar; the others label the sense
    assert entry["senses"] == ["dog, hound", "(derogatory) scoundrel — e.g. ein gemeiner Hund (a mean dog)"]
    assert entry["audio"] == [
        {"url": "https://x/De-Hund.ogg.mp3", "filename": "De-Hund.mp3", "file": "De-Hund.ogg"},
        {"url": "https://x/De-at-Hund.ogg", "filename": "De-at-Hund.ogg", "file": "De-at-Hund.ogg", "accent": "Austria"}]


def test_resolve_audio_takes_the_urls_commons_gives():
    commons = {"query": {
        "redirects": [{"from": "File:De-at-Hund.ogg", "to": "File:De-at-Hund2.ogg"}],
        "pages": {"1": {"title": "File:De-Hund.ogg", "videoinfo": [{"url": "https://u/De-Hund.ogg?utm=1"}]},
                  "2": {"title": "File:De-at-Hund2.ogg", "videoinfo": [{
                      "url": "https://u/De-at-Hund2.ogg",
                      "derivatives": [{"type": "audio/ogg", "src": "https://u/t.ogg"},
                                      {"type": "audio/mpeg", "src": "https://u/De-at-Hund2.ogg.mp3"}]}]}}}}
    entries = [compact_entry(HUND)]
    resolve_audio(FakeCache({COMMONS_API: commons}), entries)
    # The guessed mp3 does not exist: the original ogg; the renamed file: its mp3 version
    assert entries[0]["audio"] == [
        {"url": "https://u/De-Hund.ogg", "filename": "De-Hund.ogg"},
        {"url": "https://u/De-at-Hund2.ogg.mp3", "filename": "De-at-Hund2.mp3", "accent": "Austria"}]

    entries = [compact_entry(HUND)]
    resolve_audio(FakeCache({}), entries)  # Commons does not answer: the guesses stay
    assert [a["url"] for a in entries[0]["audio"]] == ["https://x/De-Hund.ogg.mp3", "https://x/De-at-Hund.ogg"]


def test_dictionary_lookup_tries_the_capitalized_word_and_adds_examples():
    cache = FakeCache({kaikki_url("Hund", "German"): json.dumps(HUND) + "\n",
                       "https://api.tatoeba.org/unstable/sentences": TATOEBA})
    result = json.loads(DictionaryLookupTool(cache)(word="hund", language="de"))
    assert result["word"] == "Hund" and result["entries"][0]["pos"] == "noun"
    assert result["examples"] == [{"text": "Ich will Hunde.", "translation": "I want dogs."}]
    assert cache.requests[-1][1]["q"] == "Hund" and cache.requests[-1][1]["trans:lang"] == "eng"

    result = json.loads(DictionaryLookupTool(cache)(word="Hund", language="German", translation_language="ru"))
    assert result["examples"] == [{"text": "Hunde rennen schnell.", "translation": "Собаки бегают быстро."}]


def test_dictionary_lookup_reports_a_missing_word_and_a_failed_tatoeba():
    cache = FakeCache({"https://api.tatoeba.org/unstable/sentences": RuntimeError("503")})
    result = json.loads(DictionaryLookupTool(cache)(word="Hundx", language="German"))
    assert result["entries"] == [] and "do not make the entry up" in result["note"]
    assert "503" in result["examples_error"]


def test_web_cache_keeps_answers_and_404s(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(url, timeout=None, params=None, **kwargs):
        calls.append(url)
        if url.endswith("missing"):
            raise NotFound(url)
        return b'{"ok": 1}'

    monkeypatch.setattr(net, "fetch", fake_fetch)
    cache = WebCache(tmp_path)
    assert cache.get_json("https://x/a", {"q": "1"}) == {"ok": 1}
    assert cache.get_json("https://x/a", {"q": "1"}) == {"ok": 1}
    assert len(calls) == 1
    cache.get_json("https://x/a", {"q": "2"})
    assert len(calls) == 2
    for _ in range(2):
        with pytest.raises(NotFound):
            cache.get_text("https://x/missing")
    assert len(calls) == 3

    old = time.time() - 30 * 24 * 3600
    for file in tmp_path.iterdir():
        os.utime(file, (old, old))
    cache.get_json("https://x/a", {"q": "1"})
    assert len(calls) == 4


def test_html_to_text():
    title, text = html_to_text(
        "<html><head><title>Hund</title><style>p{}</style></head><body>"
        "<nav>Menu</nav><div class='vector-menu-content'><div><ul><li>Deutsch</li></ul></div><p>Lang</p></div>"
        "<div role='navigation'>Jump</div><ul><li></li></ul>"
        "<h2>Noun</h2><p>Der  Hund&nbsp;bellt.<br>Laut.</p><ul><li>dog</li></ul>"
        "<script>alert(1)</script></body></html>")
    assert title == "Hund"
    assert text == "## Noun\nDer Hund bellt.\nLaut.\n- dog"


def test_web_fetch_reads_a_long_page_in_parts(monkeypatch):
    calls = []

    def fake_fetch(url, **kwargs):
        calls.append(url)
        return ("<title>T</title><p>" + "a" * 7000 + "</p>").encode()

    monkeypatch.setattr(page, "fetch", fake_fetch)
    tool = WebFetchTool()
    first = tool(url="https://example.org")
    assert first.startswith("Title: T\nCharacters 0-6000 of 7000. Call web_fetch again with start=6000")
    second = tool(url="https://example.org", start=6000)
    assert "Characters 6000-7000 of 7000." in second and "start=" not in second
    assert len(calls) == 1
    with pytest.raises(ValueError):
        tool(url="file:///c:/secret.txt")


def image_bytes(size, mode="RGB", fmt="PNG") -> bytes:
    out = BytesIO()
    Image.new(mode, size, (200, 100, 50, 128) if mode == "RGBA" else (200, 100, 50)).save(out, fmt)
    return out.getvalue()


def test_prepare_image_shrinks_big_images():
    data, name = prepare_image(image_bytes((2000, 1000)), "hund.png")
    assert name == "hund.jpg"
    assert Image.open(BytesIO(data)).size == (800, 400)

    data, name = prepare_image(image_bytes((1600, 1600), "RGBA"), "logo.webp")
    image = Image.open(BytesIO(data))
    assert name == "logo.png" and image.size == (800, 800) and image.mode == "RGBA"


def test_prepare_image_keeps_small_images_and_rejects_pages():
    small = image_bytes((300, 200), fmt="JPEG")
    assert prepare_image(small, "a.jpg") == (small, "a.jpg")
    assert prepare_image(small, "a.png") == (small, "a.jpg")
    with pytest.raises(ValueError, match="web page"):
        prepare_image(b"<!DOCTYPE html><html>Forbidden</html>", "a.jpg")
    with pytest.raises(ValueError, match="not an image"):
        prepare_image(b"\x00\x01garbage", "a.jpg")


WIKIPEDIA = {"query": {"pages": {
    "2": {"index": 2, "title": "Apple Inc.", "thumbnail": {"source": "https://t/960px-Apple_logo.svg.png?utm_source=x"},
          "original": {"source": "https://o/Apple_logo.svg?utm_source=x"}},
    "1": {"index": 1, "title": "Apple", "thumbnail": {"source": "https://t/960px-Pink_lady.jpg?utm_source=x"},
          "original": {"source": "https://o/Pink_lady.jpg"}},
}}}
COMMONS = {"query": {"pages": {"5": {"index": 1, "title": "File:Red Apple.jpg",
                                     "imageinfo": [{"thumburl": "https://t/800px-Red_Apple.jpg?utm=1"}]}}}}


def test_image_search_falls_back_to_wikimedia():
    cache = FakeCache({"https://en.wikipedia.org/w/api.php": WIKIPEDIA, "https://commons.wikimedia.org/w/api.php": COMMONS})

    class DownOpenSERP:
        def mega_image(self, **kwargs):
            raise ConnectionError("refused")

    results = WebImageSearchTool(cache, DownOpenSERP())(query="apple")
    # Sorted by search rank, SVG logos skipped, tracking parameters dropped
    assert results == [
        {"title": "Wikipedia article 'Apple'", "image_url": "https://t/960px-Pink_lady.jpg", "thumbnail_url": None},
        {"title": "Red Apple", "image_url": "https://t/800px-Red_Apple.jpg", "thumbnail_url": None},
    ]


def test_image_search_uses_openserp_when_it_answers():
    hit = SimpleNamespace(title="Dog", image=SimpleNamespace(url="https://i/dog.jpg", thumbnail=""))

    class OpenSERP:
        def mega_image(self, **kwargs):
            return SimpleNamespace(results=[hit])

    results = WebImageSearchTool(FakeCache({}), OpenSERP())(query="dog")
    assert results == [{"title": "Dog", "image_url": "https://i/dog.jpg", "thumbnail_url": None}]


def test_fetch_asks_a_busy_server_again(monkeypatch):
    import httpx

    answers = [429, 429, 200]
    sleeps = []

    def fake_fetch_once(url, timeout, params, max_bytes):
        status = answers.pop(0)
        if status != 200:
            response = httpx.Response(status, headers={"Retry-After": "120"}, request=httpx.Request("GET", url))
            raise httpx.HTTPStatusError("busy", request=response.request, response=response)
        return b"ok"

    monkeypatch.setattr(net, "_fetch_once", fake_fetch_once)
    monkeypatch.setattr(net.time, "sleep", sleeps.append)
    assert net.fetch("https://x/a.mp3") == b"ok"
    assert sleeps == [net.MAX_RETRY_DELAY, net.MAX_RETRY_DELAY]

    answers[:] = [503, 503, 503]
    with pytest.raises(RuntimeError, match="busy"):
        net.fetch("https://x/a.mp3")


def test_searcher_final_answer_ignores_none_written_as_text():
    from anki_ai_assistant.web.images import SearcherFinalAnswerTool

    tool = SearcherFinalAnswerTool()
    assert tool(word="dog", image_url="u", image_path="p.jpg", error="None") == {
        "word": "dog", "image_url": "u", "image_path": "p.jpg"}
    assert tool(word="dog", image_path="null", error="no image") == {"word": "dog", "error": "no image"}


def test_image_validator_wants_a_real_download_and_a_real_try(tmp_path):
    from types import SimpleNamespace

    import pytest
    from smolagents.memory import ToolCall

    from anki_ai_assistant.web.images import make_image_validator

    validate = make_image_validator(model=None, download_dir=tmp_path)

    def memory(*names):
        return SimpleNamespace(steps=[SimpleNamespace(tool_calls=[ToolCall(name=n, arguments={}, id=n)]) for n in names])

    # An invented path, not one download_file returned
    with pytest.raises(ValueError, match="not a downloaded file"):
        validate({"word": "dog", "image_url": "u", "image_path": "/tmp/NationalGeographic_2733106.jpg"},
                 memory("web_image_search", "download_file"))
    # Giving up before any download
    with pytest.raises(ValueError, match="Do not give up"):
        validate({"word": "dog", "error": "The image download failed"}, memory("web_image_search"))
    assert validate({"word": "dog", "error": "403"}, memory("web_image_search", "download_file"))
    assert validate({"word": "dog", "error": "nothing found"}, memory(*["web_image_search"] * 3))


def test_searcher_out_of_steps_reports_its_real_errors():
    """After max_steps the model's own words were a call written as text; the manager needs the reasons."""
    import threading
    from types import SimpleNamespace

    from anki_ai_assistant.web.images import DictManagedAgent

    class Fake(DictManagedAgent):
        def __init__(self):
            self._call_lock = threading.Lock()
            self.name = "image_searcher"
            self.prompt_templates = {"managed_agent": {"task": "{{task}}"}}
            self.memory = SimpleNamespace(steps=[SimpleNamespace(error="403 Forbidden for car.jpg"),
                                                 SimpleNamespace(error=None),
                                                 SimpleNamespace(error="Image validator rejected red_car.jpg: a toy"),
                                                 SimpleNamespace(error="Reached max steps.")])

        def run(self, task, **kwargs):
            return '(I used the tool download_file with {"url": "https://x/car.jpg"}.)'

    answer = Fake()("voiture (car)")
    assert answer["word"] == "voiture (car)" and "used all its steps" in answer["error"]
    assert "403 Forbidden" in answer["error"] and "a toy" in answer["error"] and "I used the tool" not in answer["error"]


def test_searcher_steps_and_validator_verdicts_are_logged(tmp_path):
    """The manager's log shows only the image_searcher call: its own steps and the verdicts go to the log too."""
    import logging

    from smolagents.memory import ActionStep, ToolCall
    from smolagents.monitoring import Timing

    from anki_ai_assistant.core.logs import log, log_step
    from anki_ai_assistant.web.images import build_image_searcher, make_image_validator

    records = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())
    log.addHandler(handler)
    old_level = log.level
    log.setLevel(logging.INFO)
    try:
        searcher = build_image_searcher(model=SimpleNamespace(model_id="m"), validator_model=None,
                                        download_dir=tmp_path, cache=WebCache(None), max_steps=3, verbosity_level=-1)
        step = ActionStep(step_number=2, timing=Timing(start_time=0, end_time=1),
                          tool_calls=[ToolCall(name="web_image_search", arguments={"query": "dog"}, id="1")])
        searcher.step_callbacks.callback(step, agent=searcher)
        assert records[-1].startswith("image_searcher step 2") and "web_image_search" in records[-1]
        log_step(step)
        assert records[-1].startswith("step 2")

        path = tmp_path / "dog.jpg"
        Image.new("RGB", (8, 8)).save(path)
        model = SimpleNamespace(generate=lambda *a, **k: SimpleNamespace(
            content='{"verdict": "NO", "reason": "a cat"}'))
        validate = make_image_validator(model=model, download_dir=tmp_path)
        with pytest.raises(ValueError, match="a cat"):
            validate({"word": "dog", "image_path": str(path)}, SimpleNamespace(steps=[]),
                     agent=SimpleNamespace(manager_task="Hund (dog)"))
        assert "validator NO for 'Hund (dog)'" in records[-1] and "a cat" in records[-1]
    finally:
        log.removeHandler(handler)
        log.setLevel(old_level)


def test_openserp_failure_is_logged_before_the_wikimedia_fallback():
    import logging

    from anki_ai_assistant.core.logs import log

    class DownSERP:
        def mega_image(self, **kwargs):
            raise ConnectionError("connection refused")

    records = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())
    log.addHandler(handler)
    old_level = log.level
    log.setLevel(logging.INFO)
    try:
        # The fake Wikimedia knows no URL: the fallback ran
        with pytest.raises(NotFound):
            WebImageSearchTool(FakeCache({}), DownSERP()).forward("dog")
    finally:
        log.removeHandler(handler)
        log.setLevel(old_level)
    assert any("OpenSERP image search failed for 'dog'" in r and "connection refused" in r for r in records)
