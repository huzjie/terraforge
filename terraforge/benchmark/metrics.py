"""Benchmark metrics."""
from ..utils.metrics_utils import accuracy, f1, precision, recall


def task_summary(name, preds, targets, kind="acc"):
    if kind == "acc":
        return {"task": name, "score": round(accuracy(preds, targets), 4), "n": len(targets)}
    if kind == "f1":
        from ..utils.metrics_utils import confusion
        c = confusion(preds, targets)
        return {"task": name, "score": round(f1(c), 4), "n": len(targets)}
    if kind == "err":
        # lower is better -> convert to score = 1/(1+mean_err)
        m = sum(preds) / len(preds) if preds else 0.0
        return {"task": name, "score": round(1.0 / (1.0 + m), 4), "n": len(targets)}
    if kind == "iou":
        m = sum(preds) / len(preds) if preds else 0.0
        return {"task": name, "score": round(m, 4), "n": len(targets)}
    raise ValueError(kind)


def aggregate(summaries):
    if not summaries:
        return 0.0
    return round(sum(s["score"] for s in summaries) / len(summaries), 4)
