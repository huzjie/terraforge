"""Positional embeddings: learnable and rotary (RoPE)."""
import math


def rotary_freqs(dim, max_seq=4096, base=10000.0):
    return [1.0 / (base ** (2 * (i // 2) / dim)) for i in range(dim)]


def apply_rope(x, pos, freqs):
    """Apply rotary position embedding to a vector x at position pos."""
    out = list(x)
    for i in range(0, len(x) - 1, 2):
        theta = pos * freqs[i]
        c, s = math.cos(theta), math.sin(theta)
        x0, x1 = x[i], x[i + 1]
        out[i] = x0 * c - x1 * s
        out[i + 1] = x0 * s + x1 * c
    return out


class LearnablePositionalEmbedding:
    def __init__(self, max_seq, dim, seed=0):
        import random
        rng = random.Random(seed)
        self.weight = [[rng.gauss(0, 0.02) for _ in range(dim)] for _ in range(max_seq)]

    def forward(self, seq):
        return [list(row) for row in self.weight[:seq]]

    def __call__(self, seq):
        return self.forward(seq)


def sinusoidal(max_seq, dim):
    table = []
    for pos in range(max_seq):
        row = []
        for i in range(dim):
            if i % 2 == 0:
                row.append(math.sin(pos / (10000 ** (i / dim))))
            else:
                row.append(math.cos(pos / (10000 ** ((i - 1) / dim))))
        table.append(row)
    return table
