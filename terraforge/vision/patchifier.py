"""Image -> patch sequence conversion with arbitrary-resolution support."""
from ..utils.hashing import stable_float, stable_seed
import random


def _flatten_imgs(img):
    """img is (C,H,W) nested list -> flat list of channel-major pixels."""
    out = []
    for c in range(len(img)):
        for h in range(len(img[c])):
            out.extend(img[c][h])
    return out


def adaptive_tiling(image_size, patch_size=16, max_tiles=16):
    """Choose (num_tiles_h, num_tiles_w) so any-resolution images fit max tiles."""
    h, w = image_size
    th = max(1, (h + patch_size - 1) // patch_size)
    tw = max(1, (w + patch_size - 1) // patch_size)
    # keep total tiles within budget by halving tile grid when needed
    while th * tw > max_tiles:
        if th >= tw:
            th = max(1, th // 2)
        else:
            tw = max(1, tw // 2)
    return th, tw


def patchify(img, patch_size=16):
    """Convert (C,H,W) image to a list of flattened patches.

    Returns (patches, grid) where patches is list of vectors of length C*p*p
    and grid is (th, tw).
    """
    c = len(img)
    h = len(img[0])
    w = len(img[0][0])
    th, tw = adaptive_tiling((h, w), patch_size, max_tiles=256)
    # pad the image to full patch grid with zeros
    ph, pw = th * patch_size, tw * patch_size
    patches = []
    for ty in range(th):
        for tx in range(tw):
            vec = []
            for ch in range(c):
                for py in range(patch_size):
                    for px in range(patch_size):
                        y = ty * patch_size + py
                        x = tx * patch_size + px
                        val = 0.0
                        if y < h and x < w:
                            val = float(img[ch][y][x])
                        vec.append(val)
            patches.append(vec)
    return patches, (th, tw)


def positional_patch_encode(grid, dim, seed="patch-pos"):
    """Deterministic 2D positional encoding for patch grid."""
    th, tw = grid
    enc = []
    rng = random.Random(stable_seed(seed))
    for ty in range(th):
        for tx in range(tw):
            row = [rng.gauss(0, 0.02) for _ in range(dim)]
            # inject grid coordinate signal
            row[0] += ty / max(th, 1)
            row[1] += tx / max(tw, 1)
            enc.append(row)
    return enc
