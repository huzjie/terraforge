"""Multimodal Mixture-of-Experts with load balancing."""
import math


class MultimodalMoE:
    def __init__(self, dim, num_experts=8, top_k=2, seed=0):
        from ..core.nn import Linear
        self.gate = Linear(dim, num_experts, seed=seed)
        self.num_experts = num_experts
        self.top_k = top_k
        # experts = per-expert small MLP
        from ..core.modules import MLP
        self.experts = [MLP(dim, 2 * dim, seed=seed + i) for i in range(num_experts)]

    def forward(self, x, return_aux=False):
        logits = self.gate(x)
        order = sorted(range(self.num_experts), key=lambda i: logits[i], reverse=True)
        top = order[:self.top_k]
        sel = [logits[i] for i in top]
        mx = max(sel)
        ex = [math.exp(v - mx) for v in sel]
        s = sum(ex)
        w = [e / s for e in ex]
        out = [0.0] * len(x)
        for wi, eid in zip(w, top):
            eo = self.experts[eid](x)
            for d in range(len(x)):
                out[d] += wi * eo[d]
        aux = 0.0
        if return_aux:
            p = [1.0 / self.num_experts] * self.num_experts
            g = [1.0 / (1.0 + math.exp(-logits[i])) for i in range(self.num_experts)]
            gs = sum(g) or 1.0
            g = [v / gs for v in g]
            aux = -sum(p[i] * math.log(max(g[i], 1e-9)) for i in range(self.num_experts))
        return (out, aux) if return_aux else out

    def __call__(self, x, return_aux=False):
        return self.forward(x, return_aux)
