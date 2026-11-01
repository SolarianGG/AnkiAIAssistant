**providers**: OpenAI-compatible endpoints. `api_base` must end with `/v1` (Ollama: `http://localhost:11434/v1`). The key is stored in plain text.

**roles**: which provider and model each agent uses.
- `manager`: the chat assistant that works with your collection. A cloud model is recommended.
- `searcher`: finds and downloads images for a word.
- `validator`: checks the found images, so its model must accept image input.

`max_steps` limits the agent's steps per request. `llm_timeout` is the per-request timeout in seconds.
