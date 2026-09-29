"""Higher-level modules: MLP, Transformer block, MoE layer."""
import math

from . import activations
from .attention import MultiHeadAttention
from .nn import Linear, LayerNorm


class MLP:
    def __init__(self, hidden, intermediate=None, activation="gelu", seed=0):
        self.hidden = hidden
        self.intermediate = intermediate or 4 * hidden
        self.activation = activation
        self.fc1 = Linear(hidden, self.intermediate, seed=seed)
        self.fc2 = Linear(self.intermediate, hidden, seed=seed + 1)

    def forward(self, x):
        h = self.fc1(x)
        h = [activations.apply(self.activation, v) for v in h]
        return self.fc2(h)

    def __call__(self, x):
        return self.forward(x)


class TransformerBlock:
    def __init__(self, hidden, num_heads, causal=False, sliding_window=None,
                 dropout=0.1, mlp_ratio=4, seed=0):
        self.ln1 = LayerNorm(hidden)
        self.attn = MultiHeadAttention(hidden, num_heads, causal=causal,
                                       sliding_window=sliding_window)
        self.ln2 = LayerNorm(hidden)
        self.mlp = MLP(hidden, mlp_ratio * hidden, seed=seed)
        self.dropout = dropout

    def forward(self, x):
        attn_out = self.attn(self.ln1(x))
        h = [[a + b for a, b in zip(xi, ai)] for xi, ai in zip(x, attn_out)]
        mlp_out = self.mlp(self.ln2(h))
        return [[a + b for a, b in zip(hi, mi)] for hi, mi in zip(h, mlp_out)]

    def __call__(self, x):
        return self.forward(x)


class MoELayer:
    """Sparse Mixture-of-Experts with Top-K routing and load-balancing loss."""
    def __init__(self, hidden, num_experts=8, top_k=2, seed=0):
        self.hidden = hidden
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = Linear(hidden, num_experts, seed=seed)
        self.experts = [MLP(hidden, 2 * hidden, seed=seed + i) for i in range(num_experts)]

    def forward(self, x, return_aux=False):
        logits = self.gate(x)
        order = sorted(range(self.num_experts), key=lambda i: logits[i], reverse=True)
        top = order[:self.top_k]
        # softmax over selected experts
        sel = [logits[i] for i in top]
        mx = max(sel)
        ex = [math.exp(v - mx) for v in sel]
        s = sum(ex)
        weights = [e / s for e in ex]
        out = [0.0] * self.hidden
        for w, eid in zip(weights, top):
            eo = self.experts[eid](x)
            for d in range(self.hidden):
                out[d] += w * eo[d]
        aux = None
        if return_aux:
            # simple load-balancing proxy: entropy of gate distribution
            p = [1.0 / self.num_experts] * self.num_experts
            g = [0.0] * self.num_experts
            for i in range(self.num_experts):
                g[i] = 1.0 / (1.0 + math.exp(-logits[i]))
            gs = sum(g) or 1.0
            g = [v / gs for v in g]
            aux = -sum(p[i] * math.log(max(g[i], 1e-9)) for i in range(self.num_experts))
        return (out, aux) if return_aux else out

    def __call__(self, x, return_aux=False):
        return self.forward(x, return_aux)
