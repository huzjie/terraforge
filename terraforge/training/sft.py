"""Supervised fine-tune on spatial QA pairs."""
import json

from ..backends import build_backend
from ..data.synth import SynthEngine
from ..data.annotation import qa_pairs
from ..utils.logging import get_logger

log = get_logger("terraforge.sft")


def run_sft(cfg, steps=None):
    be = build_backend(cfg.backend, cfg)
    synth = SynthEngine(seed=cfg.data.synth_seed)
    n = min(cfg.data.num_scenes, 64)
    scenes = synth.generate_many(n)

    pairs = []
    for sc in scenes:
        for qa in qa_pairs(sc):
            pairs.append((sc.id, qa["question"]))
    if not pairs:
        pairs = [(scenes[0].id, "What is left of the table?")]

    def evaluate():
        if not pairs:
            return 1.0
        c = sum(1 for sid, q in pairs if be.is_correct(sid, q))
        return c / len(pairs)

    acc_initial = evaluate()  # at skill=0.5
    skill_initial = be.skill

    steps = steps or cfg.train.max_steps
    for s in range(steps):
        sid, q = pairs[s % len(pairs)]
        info = be.train_step(sid, q)
        if s % 100 == 0 or s == steps - 1:
            log.info(f"sft step {s:4d}: correct={info['correct']} skill={info['skill']:.3f}")

    acc_final = evaluate()  # at skill=1.0
    result = {
        "backend": cfg.backend,
        "steps": steps,
        "num_pairs": len(pairs),
        "skill_initial": skill_initial,
        "skill_final": be.skill,
        "accuracy_initial": round(acc_initial, 4),
        "accuracy_final": round(acc_final, 4),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
