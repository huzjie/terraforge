"""Optional backbone abstraction (mock-friendly; real backends swap in)."""
from .encoder import VisionEncoder


def build_vision_backbone(cfg):
    return VisionEncoder(
        hidden=cfg.vision.hidden_size,
        patch_size=cfg.vision.patch_size,
        num_layers=cfg.vision.num_layers,
        num_heads=cfg.vision.num_heads,
        seed=cfg.seed,
    )
