"""vLLM backend (requires optional vllm package)."""
from .base import Backend
from .registry import register_backend


@register_backend("vllm")
class VLLMBackend(Backend):
    name = "vllm"

    def __init__(self, cfg=None):
        self.cfg = cfg
        self._llm = None

    def _ensure(self):
        if self._llm is None:
            try:
                from vllm import LLM, SamplingParams
                self._llm = LLM(model=self.cfg.model.vocab_size and "placeholder" or "placeholder")
            except ImportError as e:
                raise RuntimeError("vllm is not installed; pip install vllm") from e

    def generate(self, prompt, max_tokens=64, sample=False):
        self._ensure()
        return f"[vllm] {prompt[:40]}"

    def predict_spatial(self, scene_id, query):
        self._ensure()
        return f"[vllm] {query}"

    def embed_scene(self, scene_id):
        self._ensure()
        return [0.0] * 256
