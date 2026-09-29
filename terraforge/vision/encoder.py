"""VisionEncoder: image(s) -> global visual embedding (arbitrary resolution)."""
from .patchifier import patchify, positional_patch_encode
from .vit import ViT, PatchEmbedding
from .resolution import global_pool


class VisionEncoder:
    def __init__(self, hidden=768, patch_size=16, num_layers=6, num_heads=12, seed=0):
        self.hidden = hidden
        self.patch_size = patch_size
        self.patch_embed = PatchEmbedding(3 * patch_size * patch_size, hidden, seed=seed)
        self.vit = ViT(hidden, num_layers, num_heads, seed=seed + 1)

    def encode(self, img):
        patches, grid = patchify(img, self.patch_size)
        tokens = self.patch_embed(patches)
        pos = positional_patch_encode(grid, self.hidden)
        tokens = [[t + p for t, p in zip(tk, pk)] for tk, pk in zip(tokens, pos)]
        out = self.vit(tokens)
        return global_pool(out, "mean"), grid

    def encode_multi(self, imgs):
        """Multi-image: encode each, then mean-pool the global embeddings."""
        embs = [self.encode(im)[0] for im in imgs]
        dim = len(embs[0])
        return [sum(e[d] for e in embs) / len(embs) for d in range(dim)]

    def encode_video(self, frames):
        """Long video: encode frames and pool across time."""
        return self.encode_multi(frames)

    def __call__(self, img):
        return self.encode(img)
