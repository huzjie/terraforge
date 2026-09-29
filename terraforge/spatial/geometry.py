"""Spatial geometry primitives (no external math libs)."""
import math


def distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def relative_position(a, b):
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


def azimuth(a, b):
    """Horizontal angle (radians) from object a to b in the xy-plane."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.atan2(dy, dx)


def normalize(v):
    n = math.sqrt(sum(x * x for x in v))
    if n == 0:
        return [0.0] * len(v)
    return [x / n for x in v]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]
