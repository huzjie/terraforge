"""Attention variants: full, causal, sliding-window, and DeepSeek-style sparse.

All operate on list-of-rows representations to stay dependency-free.
"""
import math


def _softmax_rows(rows):
    out = []
    for row in rows:
        mx = max(row)
        ex = [math.exp(v - mx) for v in row]
        s = sum(ex)
        out.append([e / s for e in ex])
    return out


def _matmul(a, b):
    # a: (m,k) rows; b: (k,n) rows -> (m,n)
    m, k = len(a), len(b)
    n = len(b[0])
    out = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for p in range(k):
                acc += a[i][p] * b[p][j]
            out[i][j] = acc
    return out


def attention(q, k, v, mask=None, sparse_window=None):
    seq = len(q)
    dim = len(q[0])
    scores = [[0.0] * seq for _ in range(seq)]
    scale = 1.0 / math.sqrt(dim)
    for i in range(seq):
        qi = q[i]
        for j in range(seq):
            if sparse_window is not None and abs(i - j) > sparse_window:
                continue
            if mask is not None and not mask(i, j):
                scores[i][j] = -1e9
                continue
            acc = 0.0
            kj = k[j]
            for d in range(dim):
                acc += qi[d] * kj[d]
            scores[i][j] = acc * scale
    probs = _softmax_rows(scores)
    return _matmul(probs, v)


class MultiHeadAttention:
    def __init__(self, hidden, num_heads, causal=False, sliding_window=None):
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.causal = causal
        self.sliding_window = sliding_window
        # simple projections (deterministic init)
        self.wq = self._proj(hidden, hidden, 1)
        self.wk = self._proj(hidden, hidden, 2)
        self.wv = self._proj(hidden, hidden, 3)
        self.wo = self._proj(hidden, hidden, 4)

    @staticmethod
    def _proj(rows, cols, seed):
        import random
        rng = random.Random(seed)
        return [[rng.gauss(0, 0.02) for _ in range(cols)] for _ in range(rows)]

    def _split(self, x):
        # x: (seq, hidden) -> list of heads, each (seq, head_dim)
        heads = []
        seq = len(x)
        for h in range(self.num_heads):
            head = []
            base = h * self.head_dim
            for i in range(seq):
                head.append(list(x[i][base:base + self.head_dim]))
            heads.append(head)
        return heads

    def forward(self, x):
        seq = len(x)
        q = _matmul(x, self.wq)
        k = _matmul(x, self.wk)
        v = _matmul(x, self.wv)
        qh = self._split(q)
        kh = self._split(k)
        vh = self._split(v)
        out_heads = []
        mask = None
        if self.causal:
            mask = lambda i, j: j <= i
        for h in range(self.num_heads):
            out_heads.append(attention(qh[h], kh[h], vh[h], mask, self.sliding_window))
        # concat heads -> (seq, hidden)
        out = []
        for i in range(seq):
            row = []
            for h in range(self.num_heads):
                row.extend(out_heads[h][i])
            out.append(row)
        return _matmul(out, self.wo)

    def __call__(self, x):
        return self.forward(x)
