import unittest
from terraforge.backends import build_backend
from terraforge.spatial.world import world_truth


class TestMockBackend(unittest.TestCase):
    def setUp(self):
        self.be = build_backend("mock")

    def test_skill_monotonic(self):
        prev = self.be.skill
        for _ in range(20):
            self.be.train_step("scene-001", "What is left of the table?")
            self.assertGreaterEqual(self.be.skill, prev)
            prev = self.be.skill
        self.assertAlmostEqual(self.be.skill, 1.0, places=5)

    def test_predict_returns_string(self):
        ans = self.be.predict_spatial("scene-001", "What is left of the table?")
        self.assertIsInstance(ans, str)

    def test_embedding_dim(self):
        self.assertEqual(len(self.be.embed_scene("scene-001")), 256)

    def test_generate(self):
        out = self.be.generate("hello", max_tokens=16)
        self.assertIsInstance(out, str)


if __name__ == "__main__":
    unittest.main()
