"""LangChain LLM wrapper (optional)."""
from ..backends import build_backend


class TerraForgeLLM:
    """LangChain-compatible LLM wrapper around a terraforge backend."""

    def __init__(self, backend="mock", cfg=None):
        self.backend = build_backend(backend, cfg)

    def __call__(self, prompt, **kwargs):
        return self.backend.generate(prompt, max_tokens=kwargs.get("max_tokens", 64))

    def predict(self, prompt, **kwargs):
        return self.backend.generate(prompt, max_tokens=kwargs.get("max_tokens", 64))


def to_langchain(backend="mock", cfg=None):
    try:
        from langchain_core.language_models.llms import LLM  # noqa: F401
    except ImportError:
        raise RuntimeError("langchain_core is not installed")
    return TerraForgeLLM(backend, cfg)
