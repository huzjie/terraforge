"""Arbitrary-resolution vision encoder (patchify -> ViT -> adaptive pooling)."""
from .encoder import VisionEncoder
from .patchifier import patchify, adaptive_tiling
from .resolution import resolve_resolution

__all__ = ["VisionEncoder", "patchify", "adaptive_tiling", "resolve_resolution"]
