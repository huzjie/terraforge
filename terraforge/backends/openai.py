"""OpenAI-compatible backend (requires `openai` or `requests` + OPENAI_API_KEY)."""
import os
from .base import Backend
from .registry import register_backend


@register_backend("openai")
class OpenAICompatBackend(Backend):
    name = "openai"

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def _ensure(self):
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

    def predict_spatial(self, scene_id, query):
        self._ensure()
        return self._chat(f"Scene {scene_id}: {query}")

    def embed_scene(self, scene_id):
        self._ensure()
        return self._embed(scene_id)

    def _chat(self, prompt):
        import json
        import urllib.request
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps({"model": self.model, "messages": [{"role": "user", "content": prompt}]}).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read().decode())
        return body["choices"][0]["message"]["content"]

    def _embed(self, text):
        import json
        import urllib.request
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": "text-embedding-3-small", "input": text}).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read().decode())
        return body["data"][0]["embedding"]
