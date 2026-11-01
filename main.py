import importlib.resources
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from PIL import Image

from smolagents import CodeAgent, LiteLLMModel, Tool, ToolCallingAgent
from smolagents.agents import RunResult, populate_template
from smolagents.models import ChatMessage, MessageRole

from tools.anki_tools import anki_tools
from openserp import OpenSERP
from tools.web_tools import OpenSERPWebPageSearchTool, OpenSERPImageSearchTool, download_file_tool

DEFAULT_MODELS = {
    "ollama": "qwen3.5:9b",
    "openrouter": "nvidia/nemotron-3-ultra-550b-a55b:free",
}

IMAGES_DIR = Path("images")

SEARCHER_INSTRUCTIONS = f"""You are a web image search agent. You get ONE word or concept and must find a picture that clearly depicts it.
1. Call web_image_search with ONLY the English translation plus "photo", e.g. query='dog photo' for 'Hund (dog)'.
   Never put the foreign word into the query: it is often a false friend in English (Hund -> "hound").
   It returns a list of results with 'title', 'image_url' and 'thumbnail_url'.
   Pick a result whose title suggests ONE typical example of the thing. Skip titles with
   "types", "breeds", "chart", "collage", "set", "infographic", "meme", "vs", "list".
2. Call download_file_tool with the chosen result's image_url, saving it into the '{IMAGES_DIR.as_posix()}/' folder, e.g. filename='{IMAGES_DIR.as_posix()}/hund.jpg'. If the download fails (403, 404, hotlinking), call download_file_tool with its thumbnail_url or try the next result.
3. Only after download_file_tool has returned the saved path, call final_answer with word, image_url and that image_path, e.g. word='Hund', image_url='https://.../dog.jpg', image_path='{IMAGES_DIR.as_posix()}/hund.jpg'.
Call one tool per step and wait for its result: never call download_file_tool and final_answer in the same step.
Your answer is checked by an image validator. If it rejects the image, read the reason and try another image."""

# Replaces smolagents' default managed agent task, which asks for a free-form 3-section report
SEARCHER_TASK_TEMPLATE = """You're '{{name}}', a helper agent. Your manager submitted this task:
---
{{task}}
---
Finish by calling final_answer with word, image_url and image_path.
If you cannot find a suitable image, call final_answer with word and error="<what went wrong>" instead."""

MANAGER_TASK = """Create the Anki deck 'Test German Deck' with 2 cards for common German nouns.
Each card: Front = an image of the thing + the German word with its article (e.g. 'der Hund'), Back = the English translation.
Steps:
1. Call anki_list_note_types to check the field names of the 'Basic' note type.
2. For each word call image_searcher with the word and its English translation, e.g. result = image_searcher(task="Hund (dog)"). One word per call.
   It returns a dict, use it directly without parsing: result["image_path"] is a downloaded and validated image.
   If the dict has an "error" key instead, no image was found: then pick another word.
3. Store each image with anki_store_media(filename=..., source=result["image_path"]) and use the returned filename.
4. Add all cards with one anki_add_notes call, e.g. {'Front': '<img src="hund.jpg"><br>der Hund', 'Back': 'dog'}."""


class SearcherFinalAnswerTool(Tool):
    """Replaces the default final_answer, whose 'any' answer smolagents exposes to the model as a string:
    the model then cannot pass a dict and writes free-form text instead."""

    name = "final_answer"
    description = ("Provides the final answer: the found image, or an error if no suitable image was found. "
                   "Pass either image_url and image_path, or error.")
    inputs = {
        "word": {"type": "string", "description": "The word the image was searched for, e.g. 'Hund'."},
        "image_url": {"type": "string", "description": "URL the image was downloaded from.", "nullable": True},
        "image_path": {"type": "string", "description": "Local path of the downloaded image, e.g. 'images/hund.jpg'.",
                       "nullable": True},
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


def build_model(provider: str, model: str | None, **kwargs) -> LiteLLMModel:
    model = model or DEFAULT_MODELS[provider]
    # litellm's MCP handler (triggered by tools=) imports mcp, which is not installed.
    # A hung OpenRouter request once blocked a run for 14 minutes: time out and resend instead
    common = {"_skip_mcp_handler": True, "timeout": float(os.getenv("LLM_TIMEOUT", 180)),
              "num_retries": int(os.getenv("LLM_NUM_RETRIES", 2)), **kwargs}

    if provider == "ollama":
        return LiteLLMModel(
            model_id=f"ollama_chat/{model}",
            api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
            flatten_messages_as_text=False,
            **common,
        )

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            sys.exit("OPENROUTER_API_KEY is not set")
        return LiteLLMModel(model_id=f"openrouter/{model}", api_key=api_key,
                            flatten_messages_as_text=False,
                            **common)

    raise ValueError(f"Unknown provider: {provider}")


def searcher_prompt_templates() -> dict:
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


def make_image_validator(model: LiteLLMModel):
    """Build a final_answer check for image_searcher. It raises with a reason the agent can act upon."""

    def validate_image(final_answer: Any, memory: Any, agent: Any = None) -> bool:
        if isinstance(final_answer, str):
            try:
                final_answer = json.loads(final_answer)
            except json.JSONDecodeError:
                raise ValueError("final_answer must be a dict with keys 'word', 'image_url', 'image_path'.")
        if not isinstance(final_answer, dict) or not ({"word", "image_path"} <= final_answer.keys()
                                                      or "error" in final_answer):
            raise ValueError("final_answer must be a dict with keys 'word', 'image_url', 'image_path'.")
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
            {"type": "text", "text": f"Is the image above a good picture for a vocabulary flashcard for '{task}'?\n"
                                     "It must show ONE typical, instantly recognizable example of the thing "
                                     "as the main subject. Photos and clear drawings or clipart are both fine.\n"
                                     "Reject (NO) if it is any of:\n"
                                     "- a collage, grid, chart, infographic or diagram with several items or variants;\n"
                                     "- mostly text or labels, or a screenshot;\n"
                                     "- an unusual or specific variant instead of a typical one "
                                     "(a rare breed, a meme or parody);\n"
                                     "- the thing is tiny or hard to make out.\n"
                                     "A background, a close-up or a slightly blurred backdrop is fine.\n"
                                     "Give a short reason and the verdict YES or NO."},
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


def main():
    load_dotenv()

    client = OpenSERP(
        base_url=os.getenv("OPENSERP_BASE_URL"),
    )

    manager_provider = os.getenv("MANAGER_LLM_PROVIDER", "ollama")
    search_provider = os.getenv("SEARCH_LLM_PROVIDER", "ollama")
    validate_provider = os.getenv("VALIDATE_LLM_PROVIDER", "ollama")
    # The validator model must support image input.
    # Thinking makes Ollama reasoning models (e.g. qwen3.5) slow and cuts their answer off before the verdict,
    # litellm maps reasoning_effort="none" to Ollama's think=false
    validator_kwargs = {"reasoning_effort": "none"} if validate_provider == "ollama" else {}
    validator_model = build_model(validate_provider, os.getenv("VALIDATE_LLM_MODEL"), **validator_kwargs)

    image_searcher_agent = DictManagedAgent(
        # Calling one tool per step needs no reasoning, which only adds hidden tokens and latency
        model=build_model(search_provider, os.getenv("SEARCH_LLM_MODEL"), reasoning_effort="none"),
        tools=[OpenSERPWebPageSearchTool(client), OpenSERPImageSearchTool(client), download_file_tool,
               SearcherFinalAnswerTool()],
        name="image_searcher",
        description=(
            "Finds, downloads and validates an image for ONE word. Pass the word and its English translation, "
            "e.g. task='Hund (dog)'; call once per word. Returns a dict (not text) with 'word', 'image_url' and "
            "'image_path' (local file, ready for anki_store_media), or with 'word' and 'error'."
        ),
        instructions=SEARCHER_INSTRUCTIONS,
        prompt_templates=searcher_prompt_templates(),
        max_steps=int(os.getenv("SEARCH_MAX_STEPS", 20)),
        final_answer_checks=[make_image_validator(validator_model)],
    )
    manager_agent = CodeAgent(
        model=build_model(manager_provider, os.getenv("MANAGER_LLM_MODEL")),
        tools=anki_tools(),
        managed_agents=[image_searcher_agent],
        additional_authorized_imports=["json"],
        max_steps=int(os.getenv("MANAGER_MAX_STEPS", 20)),
        # One image_searcher call takes minutes, far beyond smolagents' default 30 s code execution limit
        executor_kwargs={"timeout_seconds": int(os.getenv("MANAGER_EXEC_TIMEOUT", 1800))},
    )

    manager_agent.run(MANAGER_TASK)


if __name__ == "__main__":
    main()
