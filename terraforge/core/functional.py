"""Functional primitives shared across modules (kept dependency-free)."""
import math


def scaled_dot_product(q, k, v, mask=None):
    """q,k,v are (seq, head_dim) lists; returns attention output (seq, head_dim)."""
    seq = len(q)
    dim = len(q[0])
    scores = [[0.0] * seq for _ in range(seq)]
    scale = 1.0 / math.sqrt(dim)
    for i in range(seq):
        for j in range(seq):
            s = 0.0
            for d in range(dim):
                s += q[i][d] * k[j][d]
            scores[i][j] = s * scale
    if mask is not None:
        for i in range(seq):
            for j in range(seq):
                if not mask(i, j):
                    scores[i][j] = -1e9
    # softmax per row
    for i in range(seq):
        row = scores[i]
        mx = max(row)
        ex = [math.exp(x - mx) for x in row]
        s = sum(ex)
        scores[i] = [e / s for e in ex]
    out = [[0.0] * dim for _ in range(seq)]
    for i in range(seq):
        for d in range(dim):
            acc = 0.0
            for j in range(seq):
                acc += scores[i][j] * v[j][d]
            out[i][d] = acc
    return out


def sliding_window_mask(window):
    def mask(i, j):
        return abs(i - j) <= window
    return mask


def causal_mask():
    def mask(i, j):
        return j <= i
    return mask


def topk(x, k):
    idx = sorted(range(len(x)), key=lambda i: x[i], reverse=True)[:k]
    return idx


def log_softmax(x):
    mx = max(x)
    ex = [math.exp(v - mx) for v in x]
    s = sum(ex)
    return [v - mx - math.log(s) for v in x]


def cross_entropy(logits, target):
    ls = log_softmax(logits)
    return -ls[target]


def accuracy(logits, targets):
    if not targets:
        return 0.0
    correct = 0
    for lg, t in zip(logits, targets):
        if max(range(len(lg)), key=lambda i: lg[i]) == t:
            correct += 1
    return correct / len(targets)
