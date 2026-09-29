"""Task 5: occlusion reasoning (in front / behind)."""
from ..metrics import task_summary
from ...spatial.world import world_truth
from ...spatial.graph import SceneGraph


def run_occlusion(backend, scenes):
    preds, targets = [], []
    for sc in scenes:
        g = SceneGraph(sc)
        for e in g.edges:
            if e.predicate not in ("in_front_of", "behind"):
                continue
            q = f"What is {e.predicate.replace('_', ' ')} the {e.object}?"
            preds.append(backend.predict_spatial(sc.id, q))
            targets.append(world_truth(sc.id, q))
    return task_summary("occlusion", preds, targets, "acc")
