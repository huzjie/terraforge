"""HuggingFace Transformers backend (requires optional transformers package)."""
from .base import Backend
from .registry import register_backend


@register_backend("transformers")
class TransformersBackend(Backend):
    name = "transformers"

    def __init__(self, cfg=None):
        self.cfg = cfg
        self._model = None

    def _ensure(self):
        if self._model is None:
            try:
                import transformers  # noqa: F401
            except ImportError as e:
                raise RuntimeError("transformers is not installed; pip install transformers") from e
            self._model = True

    def generate(self, prompt, max_tokens=64, sample=False):
        self._ensure()
        return f"[transformers] {prompt[:40]}"

    def predict_spatial(self, scene_id, query):
        self._ensure()
        return f"[transformers] {query}"

    def embed_scene(self, scene_id):
        self._ensure()
        return [0.0] * 256
