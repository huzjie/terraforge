"""Normalization helpers."""
import math


def layer_norm(x, eps=1e-5):
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    inv = 1.0 / math.sqrt(var + eps)
    return [(v - mean) * inv for v in x]


def rms_norm(x, eps=1e-6):
    ms = sum(v * v for v in x) / len(x)
    inv = 1.0 / math.sqrt(ms + eps)
    return [v * inv for v in x]
