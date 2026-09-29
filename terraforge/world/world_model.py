"""A learned world model: predict future visual+spatial state from history."""
from ..utils.hashing import stable_vector, stable_float


class WorldModel:
    def __init__(self, dim=256, seed=0):
        self.dim = dim
        self.seed = seed
        # small latent dynamics matrix (deterministic init)
        self.alpha = stable_float("world:alpha", 0.6, 0.95)

    def step(self, state, action=None):
        """Advance the latent state one step; action perturbs deterministically."""
        nxt = [self.alpha * s for s in state]
        if action is not None:
            pert = stable_vector(f"world:pert:{action}", self.dim, -0.1, 0.1)
            nxt = [n + p for n, p in zip(nxt, pert)]
        return nxt

    def predict(self, state, horizon=1, action=None):
        s = state
        for _ in range(horizon):
            s = self.step(s, action)
        return s

    def __call__(self, state, horizon=1, action=None):
        return self.predict(state, horizon, action)
