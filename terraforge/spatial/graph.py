"""Scene graph: objects as nodes, spatial relations as edges."""
from dataclasses import dataclass, field
from typing import List

from .geometry import relative_position, azimuth, distance


@dataclass
class SpatialRelation:
    subject: str
    predicate: str
    object: str
    score: float = 1.0


class SceneGraph:
    def __init__(self, scene):
        self.scene = scene
        self.nodes = [o.id for o in scene.objects]
        self.edges: List[SpatialRelation] = []
        self._build()

    def _build(self):
        for a in self.scene.objects:
            for b in self.scene.objects:
                if a.id == b.id:
                    continue
                rel = self._relate(a, b)
                if rel:
                    self.edges.append(SpatialRelation(a.id, rel, b.id))

    def _relate(self, a, b):
        rel = relative_position(a.center(), b.center())
        if rel[0] < -0.3:
            return "left_of"
        if rel[0] > 0.3:
            return "right_of"
        if rel[1] > 0.3:
            return "behind"
        if rel[1] < -0.3:
            return "in_front_of"
        if rel[2] > 0.3:
            return "above"
        if rel[2] < -0.3:
            return "below"
        return "near"

    def query(self, subject, predicate):
        return [e.object for e in self.edges
                if e.subject == subject and e.predicate == predicate]
