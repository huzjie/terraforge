import unittest
from terraforge.data.render import render_scene
from terraforge.spatial.world import build_scene
from terraforge.vision.encoder import VisionEncoder


class TestVision(unittest.TestCase):
    def test_render_shape(self):
        img = render_scene(build_scene("v-001"), H=32, W=32)
        self.assertEqual(len(img), 3)
        self.assertEqual(len(img[0]), 32)
        self.assertEqual(len(img[0][0]), 32)

    def test_encoder_embedding(self):
        enc = VisionEncoder(hidden=64, num_layers=1, num_heads=4)
        img = render_scene(build_scene("v-002"), H=32, W=32)
        emb, grid = enc.encode(img)
        self.assertEqual(len(emb), 64)


if __name__ == "__main__":
    unittest.main()
