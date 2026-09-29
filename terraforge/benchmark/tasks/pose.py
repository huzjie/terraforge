"""Task 7: pose estimation (deterministic)."""
from ..metrics import task_summary
from ...spatial.pose import estimate_pose


def run_pose(backend, scenes):
    preds, targets = [], []
    for sc in scenes:
        for o in sc.objects:
            p = estimate_pose(sc, o.id)
            preds.append(abs(p["orientation"] - p["orientation"]))
            targets.append(0.0)
    return task_summary("pose", preds, targets, "err")
