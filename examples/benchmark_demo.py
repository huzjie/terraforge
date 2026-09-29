"""Benchmark demo: run the 9-task spatial suite before/after training."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terraforge.config import load_config
from terraforge.backends import build_backend
from terraforge.benchmark.suite import BenchmarkSuite
from terraforge.data.annotation import qa_pairs
from terraforge.data.synth import SynthEngine

cfg = load_config(None)
be = build_backend("mock", cfg)

print("--- before training ---")
suite = BenchmarkSuite(be, cfg)
r0 = suite.run()
print(f"AVG before = {r0['avg']}")

# quick training
synth = SynthEngine(seed=cfg.data.synth_seed)
for s in range(300):
    sc = synth.generate_scene(f"bench-{s % 20:03d}")
    for qa in qa_pairs(sc):
        be.train_step(sc.id, qa["question"])

print("--- after training ---")
suite = BenchmarkSuite(be, cfg)
r1 = suite.run()
print(f"AVG after  = {r1['avg']}")
print(f"skill = {be.skill}")
