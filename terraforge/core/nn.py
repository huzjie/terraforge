"""Neural network building blocks (list-backed, zero-dependency)."""
import math

from . import initialization as init
from .tensor import Tensor, zeros, randn


class Linear:
    def __init__(self, in_features, out_features, bias=True, seed=0):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = init.xavier_uniform((in_features, out_features), seed)
        self.bias = [0.0] * out_features if bias else None

    def forward(self, x):
        # x: list of length in_features
        if self.bias is None:
            out = [0.0] * self.out_features
            for o in range(self.out_features):
                acc = 0.0
                for i in range(self.in_features):
                    acc += x[i] * self.weight[i * self.out_features + o]
                out[o] = acc
            return out
        out = list(self.bias)
        for o in range(self.out_features):
            acc = out[o]
            for i in range(self.in_features):
                acc += x[i] * self.weight[i * self.out_features + o]
            out[o] = acc
        return out

    def __call__(self, x):
        return self.forward(x)


class Embedding:
    def __init__(self, num_embeddings, embedding_dim, seed=0):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = init.normal((num_embeddings, embedding_dim), std=0.02, seed=seed)

    def forward(self, idx):
        base = idx * self.embedding_dim
        return list(self.weight[base:base + self.embedding_dim])

    def __call__(self, idx):
        return self.forward(idx)


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.dim = dim
        self.eps = eps
        self.gamma = [1.0] * dim
        self.beta = [0.0] * dim

    def forward(self, x):
        if x and isinstance(x[0], (list, tuple)):
            return [self.forward(row) for row in x]
        mean = sum(x) / self.dim
        var = sum((v - mean) ** 2 for v in x) / self.dim
        inv = 1.0 / math.sqrt(var + self.eps)
        return [self.gamma[i] * (x[i] - mean) * inv + self.beta[i] for i in range(self.dim)]

    def __call__(self, x):
        return self.forward(x)


class Dropout:
    def __init__(self, p=0.1, seed=0):
        self.p = p
        self.seed = seed

    def forward(self, x, train=True):
        if not train or self.p <= 0:
            return list(x)
        import random
        rng = random.Random(self.seed)
        scale = 1.0 / (1.0 - self.p)
        return [0.0 if rng.random() < self.p else v * scale for v in x]

    def __call__(self, x, train=True):
        return self.forward(x, train)


class Sequential:
    def __init__(self, *modules):
        self.modules = list(modules)

    def forward(self, x):
        for m in self.modules:
            x = m(x)
        return x

    def __call__(self, x):
        return self.forward(x)
