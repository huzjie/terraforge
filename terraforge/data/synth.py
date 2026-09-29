"""Autonomous synthetic scene generator (deterministic; delegates to world.build_scene)."""
from ..spatial.world import build_scene


class SynthEngine:
    """Produce deterministic synthetic scenes (the 'self-driving' data engine).

    Delegates to `spatial.world.build_scene` so that the data pipeline and the
    world-truth provider always construct the *same* scene for a given id.
    """

    def __init__(self, seed=42, max_objects=8):
        self.seed = seed
        self.max_objects = max_objects

    def generate_scene(self, scene_id):
        return build_scene(scene_id)

    def generate_many(self, n, prefix="scene"):
        return [self.generate_scene(f"{prefix}-{i:03d}") for i in range(n)]
