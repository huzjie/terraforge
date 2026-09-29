"""Benchmark task implementations (one function per task)."""
from .spatial_qa import run_spatial_qa
from .depth import run_depth
from .pose import run_pose
from .layout import run_layout
from .navigation import run_navigation
from .relation import run_relation
from .scene_graph import run_scene_graph
from .view_consistency import run_view_consistency
from .occlusion import run_occlusion

__all__ = [
    "run_spatial_qa", "run_depth", "run_pose", "run_layout", "run_navigation",
    "run_relation", "run_scene_graph", "run_view_consistency", "run_occlusion",
]
