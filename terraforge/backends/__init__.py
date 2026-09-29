"""Backend registry: mock / cpu / openai / vllm / transformers."""
from .base import Backend
from .registry import register_backend, list_backends, build_backend

# import concrete backends so their @register_backend decorators execute
from . import mock, cpu, openai, vllm, transformers  # noqa: F401

__all__ = ["Backend", "register_backend", "list_backends", "build_backend"]
