"""Resolution handling: normalize any input to a canonical patch grid."""
from .patchifier import adaptive_tiling


def resolve_resolution(image_size, patch_size=16, max_tiles=16):
    """Return a canonical (tiles_h, tiles_w, padded_h, padded_w) for any image."""
    th, tw = adaptive_tiling(image_size, patch_size, max_tiles)
    ph, pw = th * patch_size, tw * patch_size
    return {"tiles": (th, tw), "padded": (ph, pw), "native": image_size}


def global_pool(tokens, mode="mean"):
    """Pool a sequence of token vectors into one vector."""
    dim = len(tokens[0])
    if mode == "mean":
        return [sum(t[d] for t in tokens) / len(tokens) for d in range(dim)]
    if mode == "max":
        return [max(t[d] for t in tokens) for d in range(dim)]
    if mode == "cls":
        return list(tokens[0])
    raise ValueError(f"unknown pool mode {mode}")
