"""Configuration loading with a zero-dependency YAML fallback.

The managed runtime may not ship PyYAML; we fall back to a minimal YAML subset
parser (utils.yamlish) so the CLI works out of the box with no pip installs.
"""
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class VisionConfig:
    image_size: int = 224
    patch_size: int = 16
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    channels: int = 3
    any_resolution: bool = True
    max_tiles: int = 16


@dataclass
class ModelConfig:
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    vocab_size: int = 50257
    max_seq_len: int = 4096
    moe: bool = True
    num_experts: int = 8
    top_k: int = 2
    sliding_window: int = 512
    dropout: float = 0.1


@dataclass
class TrainConfig:
    lr: float = 3e-4
    warmup_steps: int = 100
    max_steps: int = 2000
    batch_size: int = 8
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    scheduler: str = "cosine"


@dataclass
class DataConfig:
    root: str = "data/spatial"
    num_scenes: int = 200
    synth_seed: int = 42
    train_ratio: float = 0.9


@dataclass
class ServingConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    backend: str = "mock"
    output_dir: str = "outputs"
    seed: int = 42


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _defaults() -> dict:
    return {
        "model": asdict(ModelConfig()),
        "vision": asdict(VisionConfig()),
        "train": asdict(TrainConfig()),
        "data": asdict(DataConfig()),
        "serving": asdict(ServingConfig()),
        "backend": "mock",
        "output_dir": "outputs",
        "seed": 42,
    }


def _load_yaml(path: str) -> dict:
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data
    except ImportError:
        from .utils.yamlish import parse
        with open(path, "r", encoding="utf-8") as f:
            return parse(f.read())


def load_config(path: Optional[str] = None, **overrides: Any) -> Config:
    data = _defaults()
    if path and os.path.exists(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".yaml", ".yml"):
            loaded = _load_yaml(path)
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        else:
            loaded = {}
        data = _merge(data, loaded)
    if overrides:
        data = _merge(data, overrides)
    return Config(
        model=ModelConfig(**data["model"]),
        vision=VisionConfig(**data["vision"]),
        train=TrainConfig(**data["train"]),
        data=DataConfig(**data["data"]),
        serving=ServingConfig(**data["serving"]),
        backend=data.get("backend", "mock"),
        output_dir=data.get("output_dir", "outputs"),
        seed=data.get("seed", 42),
    )


def to_dict(cfg: Config) -> Dict[str, Any]:
    return {
        "model": asdict(cfg.model),
        "vision": asdict(cfg.vision),
        "train": asdict(cfg.train),
        "data": asdict(cfg.data),
        "serving": asdict(cfg.serving),
        "backend": cfg.backend,
        "output_dir": cfg.output_dir,
        "seed": cfg.seed,
    }
