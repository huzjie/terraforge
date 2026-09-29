"""Synthetic annotation: auto-label spatial relations and depth."""
from ..spatial.graph import SceneGraph
from ..spatial.depth import estimate_depth
from ..spatial.pose import estimate_pose
from ..spatial.layout import predict_layout


def annotate_scene(scene):
    """Produce a full annotation dict for a scene."""
    graph = SceneGraph(scene)
    relations = [{"subject": e.subject, "predicate": e.predicate, "object": e.object}
                 for e in graph.edges]
    depth = {o.id: round(estimate_depth(scene, o.id), 3) for o in scene.objects}
    pose = {o.id: estimate_pose(scene, o.id) for o in scene.objects}
    layout = predict_layout(scene)
    return {
        "scene_id": scene.id,
        "objects": [{"id": o.id, "category": o.category, "x": round(o.x, 3),
                     "y": round(o.y, 3), "z": round(o.z, 3)} for o in scene.objects],
        "relations": relations,
        "depth": depth,
        "pose": {k: {"position": [round(x, 3) for x in v["position"]],
                     "orientation": round(v["orientation"], 3)} for k, v in pose.items()},
        "layout": layout,
    }


def qa_pairs(scene, graph=None):
    """Generate spatial QA pairs (question, answer) for a scene."""
    graph = graph or SceneGraph(scene)
    pairs = []
    for e in graph.edges:
        q = f"What is {e.predicate.replace('_', ' ')} the {e.object}?"
        pairs.append({"question": q, "answer": e.object, "predicate": e.predicate})
    return pairs
