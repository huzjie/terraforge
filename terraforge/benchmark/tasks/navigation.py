"""Task 9: navigation (nearest-object retrieval via near relation)."""
from ..metrics import task_summary
from ...spatial.world import world_truth


def run_navigation(backend, scenes):
    preds, targets = [], []
    for sc in scenes:
        for oid in [o.id for o in sc.objects][:6]:
            q = f"What is near the {oid}?"
            preds.append(backend.predict_spatial(sc.id, q))
            targets.append(world_truth(sc.id, q))
    return task_summary("navigation", preds, targets, "acc")
