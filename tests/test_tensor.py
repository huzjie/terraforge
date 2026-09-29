import unittest
from terraforge.core.tensor import Tensor, zeros, ones, randn, stack


class TestTensor(unittest.TestCase):
    def test_add(self):
        a = Tensor([1, 2, 3], (3,))
        b = Tensor([4, 5, 6], (3,))
        self.assertEqual((a + b).tolist(), [5, 7, 9])

    def test_matmul(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        self.assertEqual(a.matmul(b).tolist(), [[19, 22], [43, 50]])

    def test_softmax_sums_to_one(self):
        t = Tensor([1.0, 2.0, 3.0], (3,)).softmax()
        self.assertAlmostEqual(t.sum(), 1.0, places=6)

    def test_stack(self):
        a = zeros(3)
        b = ones(3)
        s = stack([a, b])
        self.assertEqual(s.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
