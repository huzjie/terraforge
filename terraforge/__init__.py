"""terraforge: spatial multimodal world model framework for the physical world.

A zero-dependency, runnable reference implementation inspired by ZDTaichu5.0-9B:
arbitrary-resolution vision encoder, spatial understanding, multimodal fusion,
world model, spatial-multimodal data production pipeline, training and serving.
"""
from .version import __version__
from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
