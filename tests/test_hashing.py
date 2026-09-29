import unittest
from terraforge.utils.hashing import stable_float, stable_ints, stable_vector, stable_seed


class TestHashing(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(stable_float("k"), stable_float("k"))
        self.assertNotEqual(stable_float("k"), stable_float("k2"))

    def test_range(self):
        for _ in range(50):
            v = stable_float("r", 0, 1)
            self.assertTrue(0.0 <= v <= 1.0)

    def test_ints_len(self):
        self.assertEqual(len(stable_ints("i", 0, 10, 7)), 7)

    def test_vector_len(self):
        self.assertEqual(len(stable_vector("v", 16)), 16)


if __name__ == "__main__":
    unittest.main()
