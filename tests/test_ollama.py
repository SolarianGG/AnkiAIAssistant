import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from anki_ai_assistant.core.config import ConfigError, load_config
from anki_ai_assistant.core.models import build_model
from anki_ai_assistant.core.ollama import model_with_context


class FakeOllama(BaseHTTPRequestHandler):
    """/api/show and /api/create of an Ollama server that has the model qwen:9b."""

    models = {}
    created = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/show":
            details = self.models.get(body["model"])
            self._reply(200, details) if details is not None else self._reply(404, {"error": "not found"})
        elif self.path == "/api/create":
            self.created.append(body)
            self.models[body["model"]] = {"model_info": {}}
            self._reply(200, {"status": "success"})
        else:
            self._reply(404, {})

    def _reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def ollama():
    FakeOllama.models = {"qwen:9b": {"model_info": {"qwen35.context_length": 65536}}}
    FakeOllama.created = []
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


def test_creates_the_derived_model_once(ollama):
    assert model_with_context(ollama, "qwen:9b", 16384) == "qwen:9b-ctx16384"
    assert FakeOllama.created == [{"model": "qwen:9b-ctx16384", "from": "qwen:9b",
                                   "parameters": {"num_ctx": 16384}, "stream": False}]
    assert model_with_context(ollama, "qwen:9b", 16384) == "qwen:9b-ctx16384"
    assert len(FakeOllama.created) == 1


def test_rejects_more_than_the_model_supports(ollama):
    with pytest.raises(ConfigError, match="65536"):
        model_with_context(ollama, "qwen:9b", 131072)


def test_unknown_model(ollama):
    with pytest.raises(ConfigError, match="ollama pull nope"):
        model_with_context(ollama, "nope", 16384)


def test_not_an_ollama_server():
    with pytest.raises(ConfigError, match="only with Ollama"):
        model_with_context("http://127.0.0.1:9/v1", "qwen:9b", 16384)


def test_build_model_uses_the_derived_model(ollama):
    config = load_config({"providers": {"ollama": {"api_base": ollama}},
                          "roles": {"validator": {"provider": "ollama", "model": "qwen:9b", "context_length": 16384}}})
    assert build_model(config, "validator").model_id == "qwen:9b-ctx16384"
    config["roles"]["validator"]["context_length"] = 0
    assert build_model(config, "validator").model_id == "qwen:9b"
