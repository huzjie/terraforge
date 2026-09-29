"""Assemble the multimodal connector (projector + fusion + router)."""
from .projector import Projector
from .fusion import GatedFusion
from .router import Router


def build_connector(cfg):
    dim = cfg.model.hidden_size
    vis_dim = cfg.vision.hidden_size
    return {
        "projector": Projector(vis_dim, dim, seed=cfg.seed),
        "fusion": GatedFusion(dim, seed=cfg.seed + 10),
        "router": Router(dim, seed=cfg.seed + 20),
    }
