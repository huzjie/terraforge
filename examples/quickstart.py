"""Quickstart: build a scene, answer a spatial query, print embeddings."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terraforge.spatial.world import build_scene, world_truth
from terraforge.backends import build_backend

be = build_backend("mock")
scene = build_scene("scene-001")
print("scene objects:", [(o.id, o.category) for o in scene.objects])
for q in ["What is left of the table?", "What is above the cup?", "nearest object to table-0"]:
    print(f"  Q: {q}")
    print(f"  A(mock)={be.predict_spatial('scene-001', q)}  truth={world_truth('scene-001', q)}")
emb = be.embed_scene("scene-001")
print("embedding dim:", len(emb))
