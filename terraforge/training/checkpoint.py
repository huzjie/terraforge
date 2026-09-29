"""Checkpoint save/load (JSON; stores skills and lightweight state)."""
from ..utils.io import write_json, read_json


def save_checkpoint(backend, path):
    state = {}
    for attr in ("_skill", "_recon_skill"):
        if hasattr(backend, attr):
            state[attr] = getattr(backend, attr)
    write_json(path, state)
    return path


def load_checkpoint(backend, path):
    state = read_json(path)
    for k, v in state.items():
        setattr(backend, k, v)
    return backend
