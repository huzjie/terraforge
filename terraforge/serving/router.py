"""Route HTTP requests to the backend."""
import json


class RequestRouter:
    def __init__(self, backend, cfg):
        self.backend = backend
        self.cfg = cfg
        self.models = ["terraforge-spatial-9b", "terraforge-mock"]

    def route(self, path, body):
        from . import openai_compat as oc
        if path == "/health":
            return {"status": "ok", "backend": self.backend.name}
        if path == "/v1/models":
            return oc.models_response(self.models)
        if path == "/v1/chat/completions":
            model, text = oc.parse_chat_request(body)
            out = self.backend.generate(text, max_tokens=128)
            return oc.chat_response(model, out)
        if path == "/v1/spatial/predict":
            sid = body.get("scene_id", "scene-001")
            q = body.get("query", "")
            ans = self.backend.predict_spatial(sid, q)
            return {"scene_id": sid, "query": q, "answer": ans}
        if path == "/v1/embeddings":
            text = body.get("input", "")
            emb = self.backend.embed_scene(text) if self.backend.name != "mock"                 else self.backend.embed_scene(text)
            return oc.embedding_response("terraforge-embed", emb)
        return {"error": "not found", "path": path}
