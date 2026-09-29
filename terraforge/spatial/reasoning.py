"""Spatial reasoning: answer relational questions against a scene graph."""
from .graph import SceneGraph
from .geometry import distance


class SpatialReasoner:
    def __init__(self, scene):
        self.scene = scene
        self.graph = SceneGraph(scene)

    def answer(self, question):
        """Parse a simple spatial question and return the deterministic answer."""
        q = question.lower()
        for pred, label in [("left of", "left_of"), ("right of", "right_of"),
                            ("above", "above"), ("below", "below"),
                            ("in front of", "in_front_of"), ("behind", "behind"),
                            ("near", "near")]:
            if pred in q:
                # extract the target object (last noun phrase before '?')
                for o in self.scene.objects:
                    if o.category.lower() in q or o.id.lower() in q:
                        result = self.graph.query(o.id, label)
                        return result[0] if result else "none"
        # fallback: nearest object to the first object
        if self.scene.objects:
            return self._nearest(self.scene.objects[0].id)
        return "unknown"

    def _nearest(self, obj_id):
        src = self.scene.get(obj_id)
        best, best_d = None, float("inf")
        for o in self.scene.objects:
            if o.id == obj_id:
                continue
            d = distance(src.center(), o.center())
            if d < best_d:
                best, best_d = o.id, d
        return best
