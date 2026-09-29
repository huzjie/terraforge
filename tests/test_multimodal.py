import unittest
from terraforge.multimodal.projector import Projector
from terraforge.multimodal.fusion import GatedFusion
from terraforge.utils.hashing import stable_vector


class TestMultimodal(unittest.TestCase):
    def test_projector(self):
        p = Projector(64, 128)
        out = p(stable_vector("v", 64))
        self.assertEqual(len(out), 128)

    def test_fusion(self):
        f = GatedFusion(64)
        t = stable_vector("t", 64)
        v = stable_vector("v", 64)
        out = f(t, v)
        self.assertEqual(len(out), 64)


if __name__ == "__main__":
    unittest.main()
