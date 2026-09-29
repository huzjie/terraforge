import unittest
from terraforge.config import Config, load_config, to_dict


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = load_config(None)
        self.assertEqual(cfg.backend, "mock")
        self.assertEqual(cfg.model.hidden_size, 768)

    def test_roundtrip(self):
        cfg = load_config(None)
        d = to_dict(cfg)
        self.assertIn("model", d)
        self.assertIn("backend", d)


if __name__ == "__main__":
    unittest.main()
