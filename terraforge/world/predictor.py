"""Next-state predictor with a learnable residual (mock-trainable)."""
from ..core.nn import Linear


class NextStatePredictor:
    def __init__(self, dim, seed=0):
        self.residual = Linear(dim, dim, seed=seed)
        self.skill = 0.5  # trainable gate, monotonic toward 1

    def forward(self, state):
        res = self.residual(state)
        return [s + self.skill * r for s, r in zip(state, res)]

    def __call__(self, state):
        return self.forward(state)
