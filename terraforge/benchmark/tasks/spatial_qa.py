"""Task 1: spatial relation question answering."""
from ..metrics import task_summary
from ...data.annotation import qa_pairs
from ...spatial.world import world_truth


def run_spatial_qa(backend, scenes):
    preds, targets = [], []
    for sc in scenes:
        for qa in qa_pairs(sc):
            preds.append(backend.predict_spatial(sc.id, qa["question"]))
            targets.append(world_truth(sc.id, qa["question"]))
    return task_summary("spatial_qa", preds, targets, "acc")
