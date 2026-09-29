"""CPU backend: run the pure-Python vision encoder + spatial reasoner."""
from ..spatial.world import build_scene
from ..spatial.reasoning import SpatialReasoner
from ..data.render import render_scene
from .base import Backend
from .registry import register_backend


@register_backend("cpu")
class CpuBackend(Backend):
    name = "cpu"

    def __init__(self, cfg=None):
        self.cfg = cfg
        from ..vision.encoder import VisionEncoder
        hidden = (cfg.vision.hidden_size if cfg else 768)
        self.encoder = VisionEncoder(hidden=hidden, num_layers=2, num_heads=8)
        self._skill = 1.0  # deterministic reasoner is exact

    def _reasoner(self, scene_id):
        return SpatialReasoner(build_scene(scene_id))

    def predict_spatial(self, scene_id, query):
        return self._reasoner(scene_id).answer(query)

    def embed_scene(self, scene_id):
        img = render_scene(build_scene(scene_id), H=32, W=32)
        emb, _ = self.encoder.encode(img)
        return emb

    def generate(self, prompt, max_tokens=64, sample=False):
        return f"[cpu] deterministic spatial answer for: {prompt[:48]}"

    @property
    def skill(self):
        return self._skill
