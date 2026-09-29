"""Learning-rate schedulers."""
import math


def cosine_schedule(lr, step, max_steps, warmup=0):
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return 0.0
    progress = (step - warmup) / max(1, max_steps - warmup)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def linear_schedule(lr, step, max_steps, warmup=0):
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    return lr * max(0.0, 1.0 - (step - warmup) / max(1, max_steps - warmup))


def constant_schedule(lr, *args, **kwargs):
    return lr
