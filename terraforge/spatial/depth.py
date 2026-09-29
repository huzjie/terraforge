"""Depth estimation (deterministic mock + interface)."""
from ..utils.hashing import stable_float, stable_vector


def estimate_depth(scene, obj_id, world_truth=None):
    """Return a pseudo-depth for an object (distance from a fixed camera origin)."""
    from .geometry import distance
    if world_truth is not None and obj_id in world_truth:
        return world_truth[obj_id]
    obj = scene.get(obj_id)
    if obj is None:
        return 0.0
    return distance((0.0, 0.0, 0.0), obj.center())


def depth_map(scene, resolution=8):
    """Produce a coarse depth grid for the scene."""
    grid = [[0.0] * resolution for _ in range(resolution)]
    for o in scene.objects:
        gx = int((o.x + 4.0) / 8.0 * resolution) % resolution
        gy = int((o.y + 4.0) / 8.0 * resolution) % resolution
        grid[gy][gx] = max(grid[gy][gx], o.z)
    return grid
