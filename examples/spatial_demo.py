"""Spatial demo: build a scene graph and reason about relations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terraforge.spatial.world import build_scene
from terraforge.spatial.graph import SceneGraph
from terraforge.spatial.reasoning import SpatialReasoner
from terraforge.spatial.depth import estimate_depth
from terraforge.spatial.layout import predict_layout

scene = build_scene("demo-001")
g = SceneGraph(scene)
print("edges:", [(e.subject, e.predicate, e.object) for e in g.edges[:8]])
r = SpatialReasoner(scene)
print("answer('left of table?'):", r.answer("What is left of the table?"))
print("depth:", {o.id: round(estimate_depth(scene, o.id), 2) for o in scene.objects})
print("layout grid:", predict_layout(scene))
