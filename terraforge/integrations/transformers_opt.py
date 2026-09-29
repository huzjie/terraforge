"""Optional transformers integration helpers (guarded imports)."""


def available():
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def load_pretrained_vision(model_name, **kwargs):
    if not available():
        raise RuntimeError("transformers not installed")
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return proc, model
