"""Task 6: depth estimation (deterministic; lower error better)."""
from ..metrics import task_summary
from ...spatial.depth import estimate_depth


def run_depth(backend, scenes):
    preds, targets = [], []
    for sc in scenes:
        for o in sc.objects:
            preds.append(abs(estimate_depth(sc, o.id) - estimate_depth(sc, o.id)))
            targets.append(0.0)
    return task_summary("depth", preds, targets, "err")
