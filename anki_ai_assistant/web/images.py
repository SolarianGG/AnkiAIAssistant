"""image_searcher: a managed agent that finds, downloads and validates a picture for one word."""

import importlib.resources
import json
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from smolagents import Model, Tool, ToolCallingAgent
from smolagents.agents import RunResult, populate_template
from smolagents.models import ChatMessage, MessageRole

from .download import DownloadFileTool

SEARCHER_INSTRUCTIONS = """You are a web image search agent. You get ONE word or concept and must find a picture that clearly depicts it.
1. Call web_image_search with ONLY the English translation plus "photo", e.g. query='dog photo' for 'Hund (dog)'.
   Never put the foreign word into the query: it is often a false friend in English (Hund -> "hound").
   It returns a list of results with 'title', 'image_url' and 'thumbnail_url'.
   Pick a result whose title suggests ONE typical example of the thing. Skip titles with
   "types", "breeds", "chart", "collage", "set", "infographic", "meme", "vs", "list".
2. Call download_file with the chosen result's image_url and a file name, e.g. filename='hund.jpg'.
   If the download fails (403, 404, hotlinking), call download_file with its thumbnail_url or try the next result.
3. Only after download_file has returned the saved path, call final_answer with word, image_url and that image_path.
Call one tool per step and wait for its result: never call download_file and final_answer in the same step.
Your answer is checked by an image validator. If it rejects the image, read the reason and try another image."""

# Replaces smolagents' default managed agent task, which asks for a free-form 3-section report
SEARCHER_TASK_TEMPLATE = """You're '{{name}}', a helper agent. Your manager submitted this task:
---
{{task}}
---
Finish by calling final_answer with word, image_url and image_path.
If you cannot find a suitable image, call final_answer with word and error="<what went wrong>" instead."""

SEARCHER_DESCRIPTION = (
    "Finds, downloads and validates an image for ONE word. Pass the word and its English translation, "
    "e.g. task='Hund (dog)'; call once per word. Returns a dict (not text) with 'word', 'image_url' and "
    "'image_path' (a local file to store in the collection media), or with 'word' and 'error'."
)


class SearcherFinalAnswerTool(Tool):
    """Replaces the default final_answer, whose 'any' answer smolagents exposes to the model as a string:
    the model then cannot pass a dict and writes free-form text instead."""

    name = "final_answer"
    description = ("Provides the final answer: the found image, or an error if no suitable image was found. "
                   "Pass either image_url and image_path, or error.")
    inputs = {
        "word": {"type": "string", "description": "The word the image was searched for, e.g. 'Hund'."},
        "image_url": {"type": "string", "description": "URL the image was downloaded from.", "nullable": True},
        "image_path": {"type": "string", "description": "Local path returned by download_file.", "nullable": True},
        "error": {"type": "string", "description": "What went wrong, if no suitable image was found.",
                  "nullable": True},
    }
    output_type = "object"

    def forward(self, word: str, image_url: str | None = None, image_path: str | None = None,
                error: str | None = None) -> dict:
        if error:
            return {"word": word, "error": error}
        return {"word": word, "image_url": image_url, "image_path": image_path}


class DictManagedAgent(ToolCallingAgent):
    """A managed agent that hands its final_answer to the manager as is.

    smolagents wraps a managed agent's answer into "Here is the final answer from your managed agent ...: {...}",
    and the manager then wastes steps parsing the dict back out of that text."""

    def __call__(self, task: str, **kwargs):
        # Kept for the image validator, which needs the word as the manager asked it
        self.manager_task = task
        full_task = populate_template(self.prompt_templates["managed_agent"]["task"],
                                      variables=dict(name=self.name, task=task))
        result = self.run(full_task, **kwargs)
        answer = result.output if isinstance(result, RunResult) else result
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except json.JSONDecodeError:
                pass
        return answer


def _searcher_prompt_templates() -> dict:
    templates = yaml.safe_load(
        importlib.resources.files("smolagents.prompts").joinpath("toolcalling_agent.yaml").read_text(encoding="utf-8"))
    templates["managed_agent"]["task"] = SEARCHER_TASK_TEMPLATE
    return templates


# Free-form answers drift into describing the image without a clear verdict; a JSON schema forces one.
# The reason goes first so the model looks at the image before it decides
VERDICT_FORMAT = {"type": "json_schema", "json_schema": {"name": "verdict", "schema": {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "verdict": {"type": "string", "enum": ["YES", "NO"]}},
    "required": ["reason", "verdict"],
}}}

VALIDATOR_PROMPT = """Is the image above a good picture for a vocabulary flashcard for '{task}'?
It must show ONE typical, instantly recognizable example of the thing as the main subject. Photos and clear drawings or clipart are both fine.
Reject (NO) if it is any of:
- a collage, grid, chart, infographic or diagram with several items or variants;
- mostly text or labels, or a screenshot;
- an unusual or specific variant instead of a typical one (a rare breed, a meme or parody);
- the thing is tiny or hard to make out.
A background, a close-up or a slightly blurred backdrop is fine.
Give a short reason and the verdict YES or NO."""

ANSWER_FORMAT_ERROR = "final_answer must be a dict with keys 'word', 'image_url', 'image_path', or 'word' and 'error'."


def make_image_validator(model: Model):
    """Build a final_answer check for image_searcher. It raises with a reason the agent can act upon."""

    def validate_image(final_answer: Any, memory: Any, agent: Any = None) -> bool:
        if isinstance(final_answer, str):
            try:
                final_answer = json.loads(final_answer)
            except json.JSONDecodeError:
                raise ValueError(ANSWER_FORMAT_ERROR)
        if not isinstance(final_answer, dict) or not ({"word", "image_path"} <= final_answer.keys()
                                                      or "error" in final_answer):
            raise ValueError(ANSWER_FORMAT_ERROR)
        if "error" in final_answer:
            # Let the searcher give up and report to the manager instead of looping until max_steps
            return True
        word, path = final_answer["word"], final_answer["image_path"]

        try:
            image = Image.open(path)
            image.load()
        except Exception as e:
            raise ValueError(f"Cannot open image '{path}' ({e}). It may be an HTML page, not an image: "
                             "download another result or its thumbnail.")
        image = image.convert("RGB")
        image.thumbnail((512, 512))

        # The manager's task holds the word as asked (e.g. 'Hund (dog)'), not as the searcher echoed it.
        # Not agent.task: that is the whole searcher prompt, and with it the model describes the image
        # instead of giving a verdict
        task = getattr(agent, "manager_task", None) or word
        response = model.generate([ChatMessage(role=MessageRole.USER, content=[
            {"type": "image", "image": image},
            {"type": "text", "text": VALIDATOR_PROMPT.format(task=task)},
        ])], response_format=VERDICT_FORMAT)
        text = response.content if isinstance(response.content, str) else str(response.content)
        try:
            answer = json.loads(text)
            verdict, reason = answer["verdict"], answer["reason"]
        except (json.JSONDecodeError, KeyError, TypeError):
            verdict, reason = "NO", f"unreadable validator answer {text.strip()[:200]!r}"
        if verdict != "YES":
            raise ValueError(f"Image validator rejected '{path}' for '{word}': {reason}. "
                             "Find and download another image that avoids this problem.")
        return True

    return validate_image


def build_image_searcher(model: Model, validator_model: Model, openserp_base_url: str, download_dir: Path,
                         max_steps: int, verbosity_level: int) -> DictManagedAgent:
    from openserp import OpenSERP

    from .search import OpenSERPImageSearchTool, OpenSERPWebPageSearchTool

    client = OpenSERP(base_url=openserp_base_url)
    return DictManagedAgent(
        model=model,
        tools=[OpenSERPWebPageSearchTool(client), OpenSERPImageSearchTool(client), DownloadFileTool(download_dir),
               SearcherFinalAnswerTool()],
        name="image_searcher",
        description=SEARCHER_DESCRIPTION,
        instructions=SEARCHER_INSTRUCTIONS,
        prompt_templates=_searcher_prompt_templates(),
        max_steps=max_steps,
        final_answer_checks=[make_image_validator(validator_model)],
        verbosity_level=verbosity_level,
    )
