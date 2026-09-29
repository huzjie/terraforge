"""Room/scene layout prediction (grid-based)."""
from ..utils.hashing import stable_ints, stable_seed
import random


def predict_layout(scene, grid=8):
    """Predict a coarse occupancy grid (1 = occupied)."""
    occ = [[0] * grid for _ in range(grid)]
    for o in scene.objects:
        gx = int((o.x + 4.0) / 8.0 * grid) % grid
        gy = int((o.y + 4.0) / 8.0 * grid) % grid
        occ[gy][gx] = 1
    return occ


def layout_iou(pred, gt):
    inter = union = 0
    for r in range(len(gt)):
        for c in range(len(gt[0])):
            p = 1 if pred[r][c] else 0
            g = 1 if gt[r][c] else 0
            if p and g:
                inter += 1
            if p or g:
                union += 1
    return inter / union if union else 0.0
