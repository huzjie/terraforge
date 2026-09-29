"""Deterministic trainable mock backend.

The mock models spatial QA as: pick the world-truth answer, but flip to a wrong
answer when a per-query deterministic draw `u` falls below a threshold that
shrinks as `skill` rises. Higher skill -> higher accuracy, monotonically.
"""
from ..utils.hashing import stable_float, stable_vector, stable_seed
from ..spatial.world import world_truth, world_truth_binary, build_scene, CATEGORIES
from .base import Backend
from .registry import register_backend


def _wrong_answer(scene_id, query, correct):
    """Deterministic wrong answer (a category different from the correct one)."""
    cands = [c for c in CATEGORIES if c != correct]
    pick = stable_float(f"wrong:{scene_id}:{query}", 0, len(cands) - 1)
    return cands[int(pick) % len(cands)]


@register_backend("mock")
class MockBackend(Backend):
    name = "mock"

    def __init__(self, cfg=None):
        self.cfg = cfg
        self._skill = 0.5          # spatial QA skill
        self._recon_skill = 0.5    # reconstruction skill (pretrain)
        self.noise = 0.5           # threshold scale for error

    # ---- spatial QA ----
    def predict_spatial(self, scene_id, query):
        correct = world_truth(scene_id, query)
        u = stable_float(f"ans:{scene_id}:{query}")
        threshold = self.noise * (1.0 - self._skill)
        if u < threshold:
            return _wrong_answer(scene_id, query, correct)
        return correct

    def is_correct(self, scene_id, query):
        return self.predict_spatial(scene_id, query) == world_truth(scene_id, query)

    # ---- reconstruction (pretrain) ----
    def reconstruct_count(self, scene_id):
        scene = build_scene(scene_id)
        true_n = len(scene)
        u = stable_float(f"recon:{scene_id}")
        threshold = self.noise * (1.0 - self._recon_skill)
        err = 0 if u >= threshold else 2  # deterministic integer error
        return max(1, true_n + err)

    def recon_error(self, scene_id):
        scene = build_scene(scene_id)
        return abs(self.reconstruct_count(scene_id) - len(scene))

    # ---- embedding ----
    def embed_scene(self, scene_id):
        return stable_vector(f"embed:{scene_id}", 256)

    # ---- generation ----
    def generate(self, prompt, max_tokens=64, sample=False):
        import random
        rng = random.Random(stable_seed(f"gen:{prompt}"))
        vocab = ("the table is left of the chair".split()
                 + ["left", "right", "above", "below", "near", "scene", "object"])
        if sample:
            return " ".join(rng.choice(vocab) for _ in range(max_tokens))
        return " ".join(vocab[i % len(vocab)] for i in range(max(1, max_tokens // 4)))

    # ---- training ----
    def train_step(self, scene_id, query):
        """Monotonic skill update: always positive, larger when correct."""
        correct = self.is_correct(scene_id, query)
        delta = 0.05 if correct else 0.03
        self._skill = min(1.0, self._skill + delta)
        return {"correct": correct, "skill": self._skill}

    def train_recon_step(self, scene_id):
        err = self.recon_error(scene_id)
        delta = 0.05 if err == 0 else 0.03
        self._recon_skill = min(1.0, self._recon_skill + delta)
        return {"error": err, "recon_skill": self._recon_skill}

    @property
    def skill(self):
        return self._skill

    @property
    def recon_skill(self):
        return self._recon_skill
