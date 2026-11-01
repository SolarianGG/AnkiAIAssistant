**providers**: OpenAI-compatible endpoints, any number of them. `api_base` must end with `/v1` (Ollama: `http://localhost:11434/v1`). The key is stored in plain text. An empty `api_key` falls back to the environment variable `<PROVIDER>_API_KEY`, e.g. `OPENROUTER_API_KEY`.

**roles**: which provider and model each agent uses.
- `manager`: the chat assistant that works with your collection. A cloud model is recommended.
- `searcher`: finds and downloads images for a word.
- `validator`: checks the found images, so its model must accept image input.

`reasoning_effort` is passed to the API as is (`none`, `low`, `medium`, `high`); leave it empty to use the model's default. With Ollama, `none` turns thinking off, which makes reasoning models like qwen3.5 several times faster.

`max_steps` limits the agent's steps per request.

`extra_body` (optional) adds fields to every request of the role. On OpenRouter it picks who serves the model: the cheapest hosts often break tool calls, so the assistant fails to act. For example `"extra_body": {"provider": {"order": ["fireworks", "together", "deepinfra"]}}` tries those hosts first; add `"allow_fallbacks": false` to use only them.

**openserp_base_url**: a self-hosted OpenSERP server used for image search. Leave it empty to disable image search.

`llm_timeout` is the per-request timeout in seconds, `llm_max_retries` how many times a failed request is resent.

`context_length` (optional, Ollama only) is the model's context window in tokens; `0` or no value uses the Ollama server's default (`OLLAMA_CONTEXT_LENGTH`, often 4096 or 8192). The assistant's instructions and tools alone take about 5500 tokens, so 16384 is the minimum for the manager and 32768 is recommended for longer dialogs; a larger context needs more memory. Ollama's OpenAI endpoint cannot set the context per request, so the add-on creates a model `<model>-ctx<N>` in Ollama once: it uses the same weights, nothing is downloaded, and it appears in `ollama list`. When the dialog outgrows the context, the assistant stops with an error instead of answering nonsense.

Changes take effect in the next dialog.
