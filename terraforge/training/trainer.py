"""Generic trainer loop (works with any backend)."""
from ..utils.logging import get_logger

log = get_logger("terraforge.train")


class Trainer:
    def __init__(self, backend, cfg=None):
        self.backend = backend
        self.cfg = cfg

    def train(self, steps, step_fn, log_every=10):
        history = []
        for s in range(steps):
            info = step_fn(s)
            history.append(info)
            if s % log_every == 0 or s == steps - 1:
                log.info(f"step {s:4d}: " + " ".join(f"{k}={v}" for k, v in info.items()))
        return history
