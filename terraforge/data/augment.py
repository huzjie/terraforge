"""Deterministic spatial augmentation."""
import random
from ..utils.hashing import stable_seed


def jitter_scene(scene, strength=0.2):
    import copy
    s = copy.deepcopy(scene)
    rng = random.Random(stable_seed(f"aug:{scene.id}"))
    for o in s.objects:
        o.x += rng.uniform(-strength, strength)
        o.y += rng.uniform(-strength, strength)
        o.z += rng.uniform(-strength, strength)
    return s


def flip_scene(scene):
    import copy
    s = copy.deepcopy(scene)
    for o in s.objects:
        o.x = -o.x
    return s
