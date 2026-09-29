"""Project visual features into the language embedding space."""
from ..core.nn import Linear, LayerNorm


class Projector:
    def __init__(self, in_dim, out_dim, activation="gelu", seed=0):
        self.fc1 = Linear(in_dim, out_dim, seed=seed)
        self.fc2 = Linear(out_dim, out_dim, seed=seed + 1)
        self.ln = LayerNorm(out_dim)
        self.activation = activation

    def forward(self, visual):
        from ..core.activations import apply
        h = self.fc1(visual)
        h = [apply(self.activation, v) for v in h]
        return self.ln(self.fc2(h))

    def __call__(self, visual):
        return self.forward(visual)
