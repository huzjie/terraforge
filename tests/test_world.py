import unittest
from terraforge.world.world_model import WorldModel
from terraforge.world.rollout import rollout
from terraforge.utils.hashing import stable_vector


class TestWorld(unittest.TestCase):
    def test_step_changes_state(self):
        wm = WorldModel(dim=8, seed=0)
        s = stable_vector("s", 8)
        self.assertNotEqual(wm.step(s), s)

    def test_rollout_length(self):
        wm = WorldModel(dim=8, seed=0)
        traj = rollout(wm, stable_vector("i", 8), ["a", "b"], horizon=4)
        self.assertEqual(len(traj), 5)


if __name__ == "__main__":
    unittest.main()
