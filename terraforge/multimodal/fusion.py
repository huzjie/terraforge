"""Multimodal fusion strategies: gated and cross-attention."""
from ..core.nn import Linear
from ..utils.hashing import stable_vector


class GatedFusion:
    """Learnable gating between text and visual embeddings."""
    def __init__(self, dim, seed=0):
        self.gate_proj = Linear(2 * dim, dim, seed=seed)

    def forward(self, text_emb, visual_emb):
        g = self.gate_proj(text_emb + visual_emb)
        # sigmoid gate
        import math
        g = [1.0 / (1.0 + math.exp(-v)) for v in g]
        return [g[i] * visual_emb[i] + (1.0 - g[i]) * text_emb[i] for i in range(len(text_emb))]

    def __call__(self, text_emb, visual_emb):
        return self.forward(text_emb, visual_emb)


class CrossAttentionFusion:
    """Cross-attention: visual tokens attend to text tokens (list-of-rows)."""
    def __init__(self, dim, seed=0):
        self.q_proj = Linear(dim, dim, seed=seed)
        self.k_proj = Linear(dim, dim, seed=seed + 1)
        self.v_proj = Linear(dim, dim, seed=seed + 2)

    def forward(self, visual_tokens, text_tokens):
        import math
        qs = [self.q_proj(t) for t in visual_tokens]
        ks = [self.k_proj(t) for t in text_tokens]
        vs = [self.v_proj(t) for t in text_tokens]
        out = []
        scale = 1.0 / math.sqrt(len(qs[0]))
        for q in qs:
            scores = [sum(q[d] * k[d] for d in range(len(q))) * scale for k in ks]
            mx = max(scores)
            ex = [math.exp(s - mx) for s in scores]
            sm = sum(ex)
            w = [e / sm for e in ex]
            out.append([sum(w[i] * vs[i][d] for i in range(len(vs))) for d in range(len(vs[0]))])
        return out

    def __call__(self, visual_tokens, text_tokens):
        return self.forward(visual_tokens, text_tokens)
