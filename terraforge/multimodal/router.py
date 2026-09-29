"""Routing between fused multimodal features and a text backbone."""
import math


class Router:
    """Decide whether a query needs visual grounding (fast/slow routing)."""
    def __init__(self, dim, seed=0):
        from ..core.nn import Linear
        self.score_proj = Linear(dim, 1, seed=seed)

    def forward(self, fused_emb):
        logit = self.score_proj(fused_emb)[0]
        p = 1.0 / (1.0 + math.exp(-logit))
        return p

    def route(self, fused_emb, threshold=0.5):
        return "visual" if self.forward(fused_emb) >= threshold else "text"

    def __call__(self, fused_emb):
        return self.forward(fused_emb)
