import unittest
from terraforge.backends import build_backend
from terraforge.benchmark.suite import BenchmarkSuite
from terraforge.config import load_config


class TestBenchmark(unittest.TestCase):
    def test_suite_runs(self):
        cfg = load_config(None)
        cfg.data.num_scenes = 10
        be = build_backend("mock", cfg)
        suite = BenchmarkSuite(be, cfg)
        r = suite.run()
        self.assertGreaterEqual(len(r["summaries"]), 8)
        self.assertTrue(0.0 <= r["avg"] <= 1.0)

    def test_improves_with_training(self):
        from terraforge.data.synth import SynthEngine
        from terraforge.data.annotation import qa_pairs
        cfg = load_config(None)
        cfg.data.num_scenes = 10
        be = build_backend("mock", cfg)
        before = BenchmarkSuite(be, cfg).run()["avg"]
        synth = SynthEngine(seed=cfg.data.synth_seed)
        for s in range(300):
            sc = synth.generate_scene(f"bi-{s % 10:03d}")
            for qa in qa_pairs(sc):
                be.train_step(sc.id, qa["question"])
        after = BenchmarkSuite(be, cfg).run()["avg"]
        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()
