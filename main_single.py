"""Single-agent variant of main.py for comparison: one CodeAgent searches, validates and adds the cards itself,
without the image_searcher managed agent."""
import os
from types import SimpleNamespace

from dotenv import load_dotenv
from openserp import OpenSERP
from smolagents import CodeAgent, tool

from main import IMAGES_DIR, build_model, make_image_validator
from tools.anki_tools import anki_tools
from tools.web_tools import OpenSERPImageSearchTool, download_file_tool

SINGLE_TASK = f"""Create the Anki deck 'Test German Deck' with 2 cards for common German nouns.
Each card: Front = an image of the thing + the German word with its article (e.g. 'der Hund'), Back = the English translation.
Steps:
1. Call anki_list_note_types to check the field names of the 'Basic' note type.
2. For each word find an image:
   - web_image_search(query=...) with ONLY the English translation plus "photo", e.g. query='dog photo'.
     Never put the German word into the query: it is often a false friend in English (Hund -> "hound").
     It returns a list of dicts with 'title', 'image_url' and 'thumbnail_url'. Skip results whose title has
     "types", "breeds", "chart", "collage", "set", "infographic", "meme", "vs", "list".
   - download_file_tool(url=..., filename='{IMAGES_DIR.as_posix()}/hund.jpg'). It raises on 403/404: then try the
     thumbnail_url or the next result.
   - validate_image(task='Hund (dog)', image_path=...). It returns 'YES' or 'NO: <reason>'. On NO try the next result.
   You can handle all words in one code block with loops and try/except.
   In this code interpreter `break` and `continue` are exceptions: inside a `try` with `except Exception` they get
   caught and the loop goes on. Keep them out of `try` blocks: set a flag inside and break after the `try`.
3. Store each image with anki_store_media(filename=..., source=<image_path>) and use the returned filename.
4. Add all cards with one anki_add_notes call, e.g. {{'Front': '<img src="hund.jpg"><br>der Hund', 'Back': 'dog'}}."""


def make_validate_image_tool(validator_model):
    check = make_image_validator(validator_model)

    @tool
    def validate_image(task: str, image_path: str) -> str:
        """Checks that a downloaded image is a good picture for a vocabulary flashcard.

        Args:
            task: The word with its English translation, e.g. 'Hund (dog)'.
            image_path: Local path of the downloaded image, e.g. 'images/hund.jpg'.

        Returns:
            'YES' if the image fits, otherwise 'NO: <reason>'.
        """
        try:
            check({"word": task, "image_path": image_path}, None, SimpleNamespace(manager_task=task))
        except ValueError as e:
            return f"NO: {e}"
        return "YES"

    return validate_image


def main():
    load_dotenv()

    client = OpenSERP(base_url=os.getenv("OPENSERP_BASE_URL"))
    validate_provider = os.getenv("VALIDATE_LLM_PROVIDER", "ollama")
    validator_kwargs = {"reasoning_effort": "none"} if validate_provider == "ollama" else {}
    validator_model = build_model(validate_provider, os.getenv("VALIDATE_LLM_MODEL"), **validator_kwargs)

    agent = CodeAgent(
        model=build_model(os.getenv("MANAGER_LLM_PROVIDER", "ollama"), os.getenv("MANAGER_LLM_MODEL")),
        tools=[OpenSERPImageSearchTool(client), download_file_tool, make_validate_image_tool(validator_model),
               *anki_tools()],
        additional_authorized_imports=["json"],
        max_steps=int(os.getenv("MANAGER_MAX_STEPS", 20)),
        # A code block that searches, downloads and validates several words runs far beyond the default 30 s
        executor_kwargs={"timeout_seconds": int(os.getenv("MANAGER_EXEC_TIMEOUT", 1800))},
    )
    agent.run(SINGLE_TASK)


if __name__ == "__main__":
    main()
