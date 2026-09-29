"""Deterministic spatial world model: ground-truth facts for scenes."""
from ..utils.hashing import stable_float, stable_ints, stable_seed
from .scene import Scene, Object3D

CATEGORIES = ["table", "chair", "cup", "book", "lamp", "plant", "box", "screen",
              "door", "window", "car", "person"]


def build_scene(scene_id, n_objects=None, seed=None):
    """Build a deterministic scene from its id."""
    n = n_objects or stable_ints(f"scene:{scene_id}:n", 3, 8, 1)[0]
    scene = Scene(id=scene_id)
    rng = stable_seed(f"scene:{scene_id}:layout")
    import random
    r = random.Random(rng)
    for i in range(n):
        cat = CATEGORIES[stable_ints(f"scene:{scene_id}:cat:{i}", 0, len(CATEGORIES) - 1, 1)[0]]
        oid = f"{cat}-{i}"
        scene.add(Object3D(
            id=oid, category=cat,
            x=r.uniform(-4, 4), y=r.uniform(-4, 4), z=r.uniform(0, 2),
            w=r.uniform(0.3, 1.5), h=r.uniform(0.3, 2.0), d=r.uniform(0.3, 1.5),
        ))
    return scene


def world_truth(scene_id, query):
    """Ground-truth answer for a spatial query, deterministic per (scene, query)."""
    scene = build_scene(scene_id)
    from .reasoning import SpatialReasoner
    return SpatialReasoner(scene).answer(query)


def world_truth_binary(scene_id, query):
    """Deterministic true/false signal for a query (used to score the mock backend)."""
    return stable_float(f"truth:{scene_id}:{query}") >= 0.5
