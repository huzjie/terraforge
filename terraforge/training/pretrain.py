"""Pretrain: reconstruction on synthetic scenes (object-count prediction)."""
import json

from ..backends import build_backend
from ..data.synth import SynthEngine
from ..utils.logging import get_logger

log = get_logger("terraforge.pretrain")


def run_pretrain(cfg, steps=None):
    be = build_backend(cfg.backend, cfg)
    if not hasattr(be, "train_recon_step"):
        log.warning(f"backend {cfg.backend} has no reconstruction training; skip")
        return {"backend": cfg.backend, "skipped": True}

    steps = steps or cfg.train.max_steps
    synth = SynthEngine(seed=cfg.data.synth_seed)
    n = min(cfg.data.num_scenes, 64)
    scenes = synth.generate_many(n)

    losses = []
    for s in range(steps):
        scene_id = scenes[s % n].id
        info = be.train_recon_step(scene_id)
        losses.append(info["error"])
        if s % 50 == 0 or s == steps - 1:
            avg = sum(losses[-50:]) / max(1, len(losses[-50:]))
            log.info(f"pretrain step {s:4d}: err={info['error']} recon_skill={info['recon_skill']:.3f} avg_err={avg:.3f}")

    result = {
        "backend": cfg.backend,
        "steps": steps,
        "recon_skill": be.recon_skill,
        "initial_err": round(sum(losses[:20]) / max(1, len(losses[:20])), 4),
        "final_err": round(sum(losses[-20:]) / max(1, len(losses[-20:])), 4),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
