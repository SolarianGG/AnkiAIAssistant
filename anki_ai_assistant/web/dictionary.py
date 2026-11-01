"""dictionary_lookup: grammar, pronunciation and meanings from Wiktionary (via kaikki.org), examples from Tatoeba."""

import json
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote

from smolagents import Tool

from .net import NotFound, WebCache

KAIKKI_URL = "https://kaikki.org/dictionary/{language}/meaning/{first}/{prefix}/{word}.jsonl"
TATOEBA_URL = "https://api.tatoeba.org/unstable/sentences"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# Language name as kaikki.org (English Wiktionary) spells it -> ISO 639-1 and Tatoeba's ISO 639-3 codes
LANGUAGES = {
    "Arabic": ("ar", "ara"), "Belarusian": ("be", "bel"), "Bulgarian": ("bg", "bul"), "Catalan": ("ca", "cat"),
    "Chinese": ("zh", "cmn"), "Czech": ("cs", "ces"), "Danish": ("da", "dan"), "Dutch": ("nl", "nld"),
    "English": ("en", "eng"), "Esperanto": ("eo", "epo"), "Finnish": ("fi", "fin"), "French": ("fr", "fra"),
    "German": ("de", "deu"), "Greek": ("el", "ell"), "Hebrew": ("he", "heb"), "Hindi": ("hi", "hin"),
    "Hungarian": ("hu", "hun"), "Indonesian": ("id", "ind"), "Italian": ("it", "ita"), "Japanese": ("ja", "jpn"),
    "Korean": ("ko", "kor"), "Latin": ("la", "lat"), "Norwegian Bokmål": ("nb", "nob"), "Persian": ("fa", "pes"),
    "Polish": ("pl", "pol"), "Portuguese": ("pt", "por"), "Romanian": ("ro", "ron"), "Russian": ("ru", "rus"),
    "Serbo-Croatian": ("sh", "hbs"), "Slovak": ("sk", "slk"), "Spanish": ("es", "spa"), "Swedish": ("sv", "swe"),
    "Turkish": ("tr", "tur"), "Ukrainian": ("uk", "ukr"), "Vietnamese": ("vi", "vie"),
}
ALIASES = {"norwegian": "Norwegian Bokmål", "mandarin": "Chinese", "farsi": "Persian", "croatian": "Serbo-Croatian",
           "serbian": "Serbo-Croatian", "bosnian": "Serbo-Croatian"}

MAX_ENTRIES = 4
MAX_SENSES = 6
MAX_GLOSS_CHARS = 160
MAX_FORMS = 10
MAX_AUDIO = 3
MAX_EXAMPLES = 3
MAX_EXAMPLE_CHARS = 150
# Relevance puts one- and two-word sentences ('Armer Hund!') first, and they show little of how the word is used
EXAMPLE_WORD_COUNT = "3-15"
# Form "tags" that mark table headers and templates, not forms of the word, and other spellings ('gehn', 'dawg')
SKIP_FORM_TAGS = {"table-tags", "inflection-template", "class", "canonical", "romanization", "alternative"}


def resolve_language(language: str) -> tuple[str, str | None]:
    """The kaikki.org name and the Tatoeba code of a language given by name or ISO code."""
    text = language.strip()
    lower = text.lower()
    for name, codes in LANGUAGES.items():
        if lower == name.lower() or lower in codes:
            return name, codes[1]
    if lower in ALIASES:
        name = ALIASES[lower]
        return name, LANGUAGES[name][1]
    # Any other language Wiktionary has, under its English name; Tatoeba's code is unknown then
    return text[:1].upper() + text[1:], None


def kaikki_url(word: str, language: str) -> str:
    return KAIKKI_URL.format(language=quote(language), first=quote(word[:1]), prefix=quote(word[:2]),
                             word=quote(word))


def _forms(entry: dict) -> list[str]:
    forms, seen = [], set()
    for form in entry.get("forms") or []:
        tags = [t for t in form.get("tags") or [] if not t.startswith("error")]
        # Full conjugation and declension tables are too long for a card: the head line already names the key forms
        if form.get("source") in ("conjugation", "declension", "inflection") or SKIP_FORM_TAGS & set(tags):
            continue
        text = f"{form.get('form', '')} ({', '.join(tags)})" if tags else form.get("form", "")
        if text and text not in seen:
            seen.add(text)
            forms.append(text)
    return forms[:MAX_FORMS]


def _senses(entry: dict) -> list[str]:
    senses = [s for s in entry.get("senses") or [] if s.get("glosses")]
    # Tags every sense has (e.g. 'masculine') are grammar, already in the head line; the rest are usage labels
    common = set.intersection(*(set(s.get("tags") or []) for s in senses)) if senses else set()
    result = []
    for sense in senses[:MAX_SENSES]:
        labels = [t for t in sense.get("tags") or [] if t not in common]
        gloss = sense["glosses"][-1]
        if len(gloss) > MAX_GLOSS_CHARS:
            gloss = gloss[:MAX_GLOSS_CHARS].rsplit(" ", 1)[0] + "…"
        text = (f"({', '.join(labels)}) " if labels else "") + gloss
        example = next((e for e in sense.get("examples") or [] if e.get("type") != "quotation"
                        and len(e.get("text", "")) <= MAX_EXAMPLE_CHARS), None)
        if example:
            translation = example.get("translation") or example.get("english")
            text += f" — e.g. {example['text']}" + (f" ({translation})" if translation else "")
        result.append(text)
    return result


def _audio(entry: dict) -> list[dict]:
    """Recordings as wiktextract names them; their URLs are guesses from the name, see resolve_audio."""
    audio = []
    for sound in entry.get("sounds") or []:
        if not sound.get("audio") or not (sound.get("mp3_url") or sound.get("ogg_url")):
            continue
        # mp3 plays everywhere, ogg not on iOS
        mp3 = sound.get("mp3_url")
        item = {"url": mp3 or sound["ogg_url"],
                "filename": sound["audio"].rsplit(".", 1)[0] + (".mp3" if mp3 else ".ogg"), "file": sound["audio"]}
        if sound.get("tags"):
            item["accent"] = ", ".join(sound["tags"])
        audio.append(item)
        if len(audio) == MAX_AUDIO:
            break
    return audio


def compact_entry(entry: dict) -> dict:
    """The part of a wiktextract entry that a card can use."""
    heads = [h["expansion"] for h in entry.get("head_templates") or [] if h.get("expansion")]
    ipa = list(dict.fromkeys(s["ipa"] for s in entry.get("sounds") or [] if s.get("ipa")))[:3]
    result = {"pos": entry.get("pos"), "head": heads[0] if heads else entry.get("word"), "ipa": ipa,
              "forms": _forms(entry), "senses": _senses(entry), "audio": _audio(entry)}
    return {k: v for k, v in result.items() if v}


def resolve_audio(cache: WebCache, entries: list[dict]) -> None:
    """Replace the guessed URLs of the recordings with the ones Commons gives.

    wiktextract builds a URL from the file name, which breaks for a renamed file (De-Stuhl.OGG is now
    De-Stuhl.ogg: 404), and not every recording has an mp3 version. If Commons does not answer,
    the guesses stay."""
    items = [a for e in entries for a in e.get("audio", [])]
    if not items:
        return
    titles = list(dict.fromkeys("File:" + a["file"] for a in items))
    try:
        answer = cache.get_json(COMMONS_API, params={
            "action": "query", "format": "json", "redirects": 1, "titles": "|".join(titles),
            "prop": "videoinfo", "viprop": "url|derivatives"})
    except Exception:
        for a in items:
            del a["file"]
        return
    query = answer.get("query") or {}
    renamed = {r["from"]: r["to"] for r in query.get("normalized", []) + query.get("redirects", [])}
    files = {}
    for page in (query.get("pages") or {}).values():
        info = (page.get("videoinfo") or [{}])[0]
        if "missing" in page or not info.get("url"):
            continue
        mp3 = next((d["src"] for d in info.get("derivatives") or [] if d.get("type") == "audio/mpeg"), None)
        files[page["title"]] = mp3 or info["url"].split("?", 1)[0]
    for entry in entries:
        resolved = []
        for a in entry.get("audio", []):
            title = "File:" + a.pop("file")
            for _ in range(2):  # normalized, then redirected
                title = renamed.get(title, title)
            if title in files:
                a["url"] = files[title]
                a["filename"] = PurePosixPath(title.removeprefix("File:")).stem + PurePosixPath(a["url"]).suffix.lower()
                resolved.append(a)
        if resolved:
            entry["audio"] = resolved
        else:
            entry.pop("audio", None)


def lookup_entries(cache: WebCache, word: str, language: str) -> tuple[str, list[dict]]:
    """Entries for the word as written, or else in lower case or capitalized (German nouns). Raises NotFound."""
    for variant in dict.fromkeys([word, word.lower(), word[:1].upper() + word[1:]]):
        try:
            text = cache.get_text(kaikki_url(variant, language))
        except NotFound:
            continue
        entries = [compact_entry(json.loads(line)) for line in text.splitlines()[:MAX_ENTRIES] if line.strip()]
        if entries:
            resolve_audio(cache, entries)
            return variant, entries
    raise NotFound(word)


def tatoeba_examples(cache: WebCache, word: str, language: str, translation: str) -> list[dict]:
    answer = cache.get_json(TATOEBA_URL, params={"lang": language, "q": word, "trans:lang": translation,
                                                 "sort": "relevance", "word_count": EXAMPLE_WORD_COUNT, "limit": 10})
    examples = []
    for sentence in answer.get("data") or []:
        text = sentence.get("text", "")
        if sentence.get("is_unapproved") or len(text) > MAX_EXAMPLE_CHARS:
            continue
        translations = [t for t in sentence.get("translations") or [] if t.get("lang") == translation]
        translations.sort(key=lambda t: not t.get("is_direct"))
        if translations:
            examples.append({"text": sentence["text"], "translation": translations[0]["text"]})
        if len(examples) == MAX_EXAMPLES:
            break
    return examples


class DictionaryLookupTool(Tool):
    name = "dictionary_lookup"
    risk = "read"
    description = (
        "Look a word up in Wiktionary: part of speech, a head line with gender and key forms (plural, past tense...), "
        "IPA, meanings in English, and pronunciation recordings ('audio': url and filename to pass to store_media). "
        "Also returns example sentences with translations from Tatoeba. Use it for grammar, pronunciation and "
        "examples instead of writing them from memory. Pass the base form (infinitive, nominative singular)."
    )
    inputs = {
        "word": {"type": "string", "description": "The word, e.g. 'Hund'."},
        "language": {"type": "string", "description": "Language of the word: English name or ISO code, e.g. 'German'."},
        "translation_language": {"type": "string", "nullable": True,
                                 "description": "Language for the translations of the examples (default: English)."},
    }
    output_type = "string"

    def __init__(self, cache: WebCache, **kwargs):
        super().__init__(**kwargs)
        self.cache = cache

    def forward(self, word: str, language: str, translation_language: str | None = None) -> str:
        word = word.strip()
        if not word:
            raise ValueError("Pass a word.")
        name, code = resolve_language(language)
        result: dict[str, Any] = {"word": word, "language": name}
        try:
            found, entries = lookup_entries(self.cache, word, name)
            result["word"], result["entries"] = found, entries
        except NotFound:
            result["entries"] = []
            result["note"] = (f"Wiktionary has no {name} entry for '{word}'. Check the spelling, the base form and "
                              "the language name; do not make the entry up.")

        _, translation_code = resolve_language(translation_language or "English")
        if code and translation_code and code != translation_code:
            try:
                result["examples"] = tatoeba_examples(self.cache, result["word"], code, translation_code)
            except Exception as e:  # Examples are a bonus: the entry is still useful without them
                result["examples_error"] = f"Tatoeba failed: {e}"
        return json.dumps(result, ensure_ascii=False)
