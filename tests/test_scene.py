import unittest
from terraforge.spatial.world import build_scene
from terraforge.spatial.graph import SceneGraph
from terraforge.spatial.scene import Scene, Object3D


class TestScene(unittest.TestCase):
    def test_build_scene(self):
        s = build_scene("t-001")
        self.assertTrue(3 <= len(s) <= 8)

    def test_graph_edges(self):
        s = build_scene("t-002")
        g = SceneGraph(s)
        self.assertTrue(len(g.edges) > 0)

    def test_manual_scene(self):
        s = Scene("x")
        s.add(Object3D("a", "table", 0, 0, 0))
        s.add(Object3D("b", "cup", 1, 0, 0))
        self.assertEqual(len(s), 2)


if __name__ == "__main__":
    unittest.main()
