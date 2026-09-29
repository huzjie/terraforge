"""Backend registration and construction."""
_BACKENDS = {}


def register_backend(name):
    def deco(cls):
        _BACKENDS[name] = cls
        return cls
    # allow @register_backend("name") without parentheses
    if isinstance(name, type):
        cls = name
        _BACKENDS[cls.name] = cls
        return cls
    return deco


def list_backends():
    return list(_BACKENDS.keys())


def build_backend(name, cfg=None):
    if name not in _BACKENDS:
        raise ValueError(f"unknown backend {name!r}, available: {list(_BACKENDS)}")
    return _BACKENDS[name](cfg)
