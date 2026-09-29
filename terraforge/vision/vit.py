"""A small Vision Transformer operating on flattened patches."""
import math

from ..core.nn import Linear, LayerNorm
from ..core.attention import MultiHeadAttention


class PatchEmbedding:
    def __init__(self, patch_dim, hidden, seed=0):
        self.proj = Linear(patch_dim, hidden, seed=seed)

    def forward(self, patches):
        return [self.proj(p) for p in patches]

    def __call__(self, patches):
        return self.forward(patches)


class ViT:
    def __init__(self, hidden=768, num_layers=6, num_heads=12, dropout=0.1, seed=0):
        self.hidden = hidden
        self.blocks = []
        for i in range(num_layers):
            self.blocks.append(_Block(hidden, num_heads, dropout, seed + i))

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

    def __call__(self, x):
        return self.forward(x)


class _Block:
    def __init__(self, hidden, num_heads, dropout, seed):
        self.ln1 = LayerNorm(hidden)
        self.attn = MultiHeadAttention(hidden, num_heads)
        self.ln2 = LayerNorm(hidden)
        self.mlp = _MLP(hidden, seed)

    def forward(self, x):
        a = self.attn(self.ln1(x))
        h = [[xi + ai for xi, ai in zip(r1, r2)] for r1, r2 in zip(x, a)]
        m = self.mlp(self.ln2(h))
        return [[hi + mi for hi, mi in zip(r1, r2)] for r1, r2 in zip(h, m)]

    def __call__(self, x):
        return self.forward(x)


class _MLP:
    def __init__(self, hidden, seed):
        self.fc1 = Linear(hidden, 4 * hidden, seed=seed)
        self.fc2 = Linear(4 * hidden, hidden, seed=seed + 1)

    def forward(self, x):
        return [self.fc2([max(0.0, v) for v in self.fc1(r)]) for r in x]

    def __call__(self, x):
        return self.forward(x)
