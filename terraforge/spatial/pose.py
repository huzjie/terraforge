"""Pose estimation (position + orientation, mock deterministic)."""
from ..utils.hashing import stable_vector, stable_float


def estimate_pose(scene, obj_id):
    obj = scene.get(obj_id)
    if obj is None:
        return {"position": (0.0, 0.0, 0.0), "orientation": 0.0}
    yaw = stable_float(f"pose:yaw:{scene.id}:{obj_id}", -3.14159, 3.14159)
    return {"position": obj.center(), "orientation": yaw}


def pose_error(pred, gt):
    from .geometry import distance
    pos_err = distance(pred["position"], gt["position"])
    ori_err = abs(pred["orientation"] - gt["orientation"])
    return {"position": pos_err, "orientation": ori_err, "total": pos_err + ori_err}
