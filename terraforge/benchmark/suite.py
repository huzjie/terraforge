"""BenchmarkSuite: run all 9 tasks, print a report, return AVG."""
import json

from ..backends import build_backend
from ..data.synth import SynthEngine
from ..utils.logging import get_logger
from . import tasks
from .metrics import aggregate

log = get_logger("terraforge.bench")

TASK_NAMES = [
    "spatial_qa", "relation", "scene_graph", "view_consistency", "occlusion",
    "depth", "pose", "layout", "navigation",
]


class BenchmarkSuite:
    def __init__(self, backend, cfg=None):
        self.backend = backend
        self.cfg = cfg
        self.synth = SynthEngine(seed=(cfg.data.synth_seed if cfg else 42))
        self.scenes = self.synth.generate_many(min((cfg.data.num_scenes if cfg else 200), 40))

    def run(self):
        summaries = []
        for name in TASK_NAMES:
            fn = getattr(tasks, f"run_{name}", None)
            if fn is None:
                continue
            s = fn(self.backend, self.scenes)
            summaries.append(s)
        avg = aggregate(summaries)
        return {"summaries": summaries, "avg": avg}


def run_benchmark(cfg):
    be = build_backend(cfg.backend, cfg)
    suite = BenchmarkSuite(be, cfg)
    result = suite.run()
    print("=" * 60)
    print("terraforge spatial understanding benchmark (9 tasks)")
    print("=" * 60)
    for s in result["summaries"]:
        print(f"  {s['task']:<18} score={s['score']:.4f}  (n={s['n']})")
    print("-" * 60)
    print(f"  {'AVG':<18} score={result['avg']:.4f}")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
