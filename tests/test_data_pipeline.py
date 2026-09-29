import unittest
from terraforge.data.synth import SynthEngine
from terraforge.data.render import render_scene
from terraforge.data.annotation import annotate_scene, qa_pairs


class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.synth = SynthEngine(seed=42)

    def test_generate(self):
        sc = self.synth.generate_scene("d-001")
        self.assertTrue(len(sc) >= 3)

    def test_annotate(self):
        sc = self.synth.generate_scene("d-002")
        ann = annotate_scene(sc)
        self.assertEqual(ann["scene_id"], "d-002")
        self.assertIn("relations", ann)

    def test_qa_pairs(self):
        sc = self.synth.generate_scene("d-003")
        pairs = qa_pairs(sc)
        self.assertTrue(len(pairs) > 0)
        for p in pairs:
            self.assertIn("question", p)
            self.assertIn("answer", p)


if __name__ == "__main__":
    unittest.main()
