"""Task 3: scene graph edge recall (depth relation 'behind')."""
from ..metrics import task_summary
from ...spatial.world import world_truth
from ...spatial.graph import SceneGraph


def run_scene_graph(backend, scenes):
    preds, targets = [], []
    for sc in scenes:
        g = SceneGraph(sc)
        for e in g.edges:
            if e.predicate != "behind":
                continue
            q = f"What is behind the {e.object}?"
            preds.append(backend.predict_spatial(sc.id, q))
            targets.append(world_truth(sc.id, q))
    return task_summary("scene_graph", preds, targets, "acc")
