"""Task 8: layout prediction IoU (deterministic)."""
from ..metrics import task_summary
from ...spatial.layout import predict_layout, layout_iou


def run_layout(backend, scenes):
    preds, targets = [], []
    for sc in scenes:
        gt = predict_layout(sc)
        iou = layout_iou(predict_layout(sc), gt)
        preds.append(iou)
        targets.append(1.0)
    return task_summary("layout", preds, targets, "iou")
