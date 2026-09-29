"""Activation functions (as plain functions + registry)."""
import math

ACTIVATIONS = {}


def register(name):
    def deco(fn):
        ACTIVATIONS[name] = fn
        return fn
    return deco


@register("relu")
def relu(x):
    return max(0.0, x)


@register("gelu")
def gelu(x):
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + math.tanh(c * (x + 0.044715 * x ** 3)))


@register("silu")
def silu(x):
    return x / (1.0 + math.exp(-x))


@register("tanh")
def tanh(x):
    return math.tanh(x)


@register("identity")
def identity(x):
    return x


def apply(name, x):
    fn = ACTIVATIONS.get(name)
    if fn is None:
        raise ValueError(f"unknown activation {name!r}, available: {list(ACTIVATIONS)}")
    return fn(x)
