"""Deterministic hashing helpers.

md5(key) is used to seed a random.Random for speed and entropy. The first 8
bytes of the digest drive the generator (16 bytes padded flat only yields a
handful of distinct values, which breaks downstream splits).
"""
import hashlib
import random


def _seed_bytes(key):
    if isinstance(key, str):
        key = key.encode("utf-8")
    return hashlib.md5(key).digest()[:8]


def stable_seed(key):
    return int.from_bytes(_seed_bytes(key), "big")


def stable_float(key, lo=0.0, hi=1.0):
    rng = random.Random(stable_seed(key))
    return rng.uniform(lo, hi)


def stable_ints(key, lo, hi, n):
    rng = random.Random(stable_seed(key))
    return [rng.randint(lo, hi) for _ in range(n)]


def stable_vector(key, dim, lo=-1.0, hi=1.0):
    rng = random.Random(stable_seed(key))
    return [rng.uniform(lo, hi) for _ in range(dim)]


def stable_gauss(key, dim, std=1.0):
    rng = random.Random(stable_seed(key))
    return [rng.gauss(0.0, std) for _ in range(dim)]
