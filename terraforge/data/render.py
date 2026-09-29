"""Render a synthetic scene into an image-like tensor (C,H,W)."""
from ..utils.hashing import stable_float, stable_seed
import random


def render_scene(scene, H=64, W=64, C=3):
    """Rasterize objects into a coarse RGB-ish image (deterministic per scene)."""
    img = [[[0.0] * W for _ in range(H)] for _ in range(C)]
    for o in scene.objects:
        color = [stable_float(f"color:{scene.id}:{o.id}:{c}") for c in range(C)]
        cx = int((o.x + 4.0) / 8.0 * W) % W
        cy = int((o.y + 4.0) / 8.0 * H) % H
        r = max(1, int(o.w * 2))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < W and 0 <= y < H:
                    for c in range(C):
                        img[c][y][x] = max(img[c][y][x], color[c])
    return img


def render_video(scene, n_frames=4):
    """Render a short video by slightly shifting object positions each frame."""
    frames = []
    for f in range(n_frames):
        import copy
        shifted = copy.deepcopy(scene)
        for o in shifted.objects:
            o.x += f * 0.1
        frames.append(render_scene(shifted))
    return frames
