"""Deterministic weight initialization."""
import math
import random


def xavier_uniform(shape, seed=0):
    rng = random.Random(seed)
    n = 1
    for d in shape:
        n *= d
    fan_in = shape[0] if len(shape) >= 1 else 1
    limit = math.sqrt(6.0 / fan_in) if fan_in else 0.0
    return [rng.uniform(-limit, limit) for _ in range(n)]


def kaiming_uniform(shape, seed=0):
    rng = random.Random(seed)
    n = 1
    for d in shape:
        n *= d
    fan_in = shape[0] if len(shape) >= 1 else 1
    gain = math.sqrt(2.0 / (1 + 0))  # relu gain
    std = gain / math.sqrt(fan_in) if fan_in else 1.0
    bound = math.sqrt(3.0) * std
    return [rng.uniform(-bound, bound) for _ in range(n)]


def normal(shape, std=0.02, seed=0):
    rng = random.Random(seed)
    n = 1
    for d in shape:
        n *= d
    return [rng.gauss(0.0, std) for _ in range(n)]
