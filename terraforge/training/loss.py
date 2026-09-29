"""Loss functions."""
import math


def mse(pred, target):
    return sum((p - t) ** 2 for p, t in zip(pred, target)) / len(target)


def cross_entropy(logits, target):
    mx = max(logits)
    ex = [math.exp(v - mx) for v in logits]
    s = sum(ex)
    return -logits[target] + mx + math.log(s)


def binary_cross_entropy(pred, target):
    eps = 1e-7
    pred = max(eps, min(1 - eps, pred))
    return -(target * math.log(pred) + (1 - target) * math.log(1 - pred))
