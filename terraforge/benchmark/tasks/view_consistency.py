"""Task 4: view consistency (front relation 'in_front_of')."""
from ..metrics import task_summary
from ...spatial.world import world_truth
from ...spatial.graph import SceneGraph


def run_view_consistency(backend, scenes):
    preds, targets = [], []
    for sc in scenes:
        g = SceneGraph(sc)
        for e in g.edges:
            if e.predicate != "in_front_of":
                continue
            q = f"What is in front of the {e.object}?"
            preds.append(backend.predict_spatial(sc.id, q))
            targets.append(world_truth(sc.id, q))
    return task_summary("view_consistency", preds, targets, "acc")
