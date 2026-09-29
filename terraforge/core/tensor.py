"""A minimal but real zero-dependency tensor.

Supports the subset needed to run a small spatial world model end-to-end:
1D vectors and 2D matrices, elementwise ops, matmul, softmax, layernorm,
reductions and simple indexing. Values are Python floats in a flat list.
"""
import math
import random


class Tensor:
    __slots__ = ("shape", "data")

    def __init__(self, data, shape=None):
        if shape is None:
            if isinstance(data, Tensor):
                self.shape = tuple(data.shape)
                self.data = list(data.data)
            elif isinstance(data, (list, tuple)):
                flat, shp = self._flatten(data)
                self.shape = tuple(shp)
                self.data = flat
            else:
                self.shape = ()
                self.data = [float(data)]
        else:
            self.shape = tuple(shape)
            n = self._numel()
            if isinstance(data, (list, tuple)):
                self.data = [float(x) for x in data]
            else:
                self.data = [float(data)] * n

    @staticmethod
    def _flatten(x):
        if isinstance(x, (int, float)):
            return [float(x)], ()
        if not x:
            return [], (0,)
        if isinstance(x[0], (list, tuple)):
            rows = [Tensor._flatten(r)[0] for r in x]
            ncols = len(rows[0])
            flat = []
            for r in rows:
                flat.extend(r)
            return flat, (len(rows), ncols)
        return [float(v) for v in x], (len(x),)

    def _numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def __repr__(self):
        return f"Tensor(shape={self.shape})"

    def __len__(self):
        return self.shape[0] if self.shape else 0

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if len(self.shape) == 1:
                return self.data[idx]
            if len(self.shape) == 2:
                row = self.shape[1]
                return Tensor(self.data[idx * row:(idx + 1) * row], (row,))
        raise NotImplementedError("only integer indexing supported")

    def _binary(self, other, op):
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError(f"shape mismatch {self.shape} vs {other.shape}")
            return Tensor([op(a, b) for a, b in zip(self.data, other.data)], self.shape)
        if isinstance(other, (int, float)):
            return Tensor([op(a, other) for a in self.data], self.shape)
        raise TypeError(f"unsupported operand {type(other)}")

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self._binary(other, lambda a, b: a / b if b != 0 else 0.0)

    def __neg__(self):
        return Tensor([-a for a in self.data], self.shape)

    def matmul(self, other):
        if len(self.shape) == 2 and len(other.shape) == 2:
            m, k = self.shape
            k2, n = other.shape
            if k != k2:
                raise ValueError(f"matmul shape mismatch {self.shape} x {other.shape}")
            out = [0.0] * (m * n)
            for i in range(m):
                base_i = i * k
                for j in range(n):
                    acc = 0.0
                    for p in range(k):
                        acc += self.data[base_i + p] * other.data[p * n + j]
                    out[i * n + j] = acc
            return Tensor(out, (m, n))
        if len(self.shape) == 2 and len(other.shape) == 1:
            m, k = self.shape
            out = [0.0] * m
            for i in range(m):
                acc = 0.0
                base = i * k
                for p in range(k):
                    acc += self.data[base + p] * other.data[p]
                out[i] = acc
            return Tensor(out, (m,))
        if len(self.shape) == 1 and len(other.shape) == 1:
            return Tensor(sum(a * b for a, b in zip(self.data, other.data)), ())
        raise ValueError(f"matmul unsupported shapes {self.shape} x {other.shape}")

    def transpose(self):
        if len(self.shape) != 2:
            return Tensor(self.data, self.shape)
        m, n = self.shape
        out = [0.0] * (m * n)
        for i in range(m):
            for j in range(n):
                out[j * m + i] = self.data[i * n + j]
        return Tensor(out, (n, m))

    def softmax(self, axis=-1):
        if len(self.shape) == 1:
            mx = max(self.data)
            ex = [math.exp(x - mx) for x in self.data]
            s = sum(ex)
            return Tensor([e / s for e in ex], self.shape)
        if len(self.shape) == 2 and axis in (-1, 1):
            m, n = self.shape
            out = []
            for i in range(m):
                row = self.data[i * n:(i + 1) * n]
                mx = max(row)
                ex = [math.exp(x - mx) for x in row]
                s = sum(ex)
                out.extend([e / s for e in ex])
            return Tensor(out, self.shape)
        raise NotImplementedError("softmax only for 1D/2D")

    def relu(self):
        return Tensor([max(0.0, x) for x in self.data], self.shape)

    def gelu(self):
        c = math.sqrt(2.0 / math.pi)
        out = []
        for x in self.data:
            out.append(0.5 * x * (1.0 + math.tanh(c * (x + 0.044715 * x ** 3))))
        return Tensor(out, self.shape)

    def silu(self):
        out = [x / (1.0 + math.exp(-x)) for x in self.data]
        return Tensor(out, self.shape)

    def tanh(self):
        return Tensor([math.tanh(x) for x in self.data], self.shape)

    def layernorm(self, eps=1e-5):
        mean = sum(self.data) / len(self.data)
        var = sum((x - mean) ** 2 for x in self.data) / len(self.data)
        inv = 1.0 / math.sqrt(var + eps)
        return Tensor([(x - mean) * inv for x in self.data], self.shape)

    def mean(self):
        return sum(self.data) / len(self.data)

    def sum(self):
        return sum(self.data)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        if -1 in shape:
            rest = self._numel() // abs(shape[0] * (shape[1] if len(shape) > 1 else 1)) if False else None
            known = 1
            neg_idx = None
            for i, d in enumerate(shape):
                if d == -1:
                    neg_idx = i
                else:
                    known *= d
            assert self._numel() % known == 0, "reshape -1 not divisible"
            shape = list(shape)
            shape[neg_idx] = self._numel() // known
            shape = tuple(shape)
        assert self._numel() == _prod(shape), f"reshape size mismatch {self._numel()} -> {shape}"
        return Tensor(list(self.data), shape)

    def tolist(self):
        if not self.shape:
            return self.data[0] if self.data else None
        if len(self.shape) == 1:
            return list(self.data)
        if len(self.shape) == 2:
            m, n = self.shape
            return [self.data[i * n:(i + 1) * n] for i in range(m)]
        return list(self.data)

    def argmax(self):
        return max(range(len(self.data)), key=lambda i: self.data[i])


def _prod(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def zeros(*shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor([0.0] * _prod(shape), shape)


def ones(*shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor([1.0] * _prod(shape), shape)


def randn(*shape, seed=None):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    rng = random.Random(seed)
    return Tensor([rng.gauss(0.0, 1.0) for _ in range(_prod(shape))], shape)


def from_list(data):
    return Tensor(data)


def stack(tensors, axis=0):
    if not tensors:
        raise ValueError("empty stack")
    s0 = tensors[0].shape
    for t in tensors:
        assert t.shape == s0, "stack shapes must match"
    if axis == 0:
        return Tensor([x for t in tensors for x in t.data], (len(tensors),) + s0)
    if axis == 1 and len(s0) == 1:
        n = s0[0]
        out = []
        for i in range(n):
            for t in tensors:
                out.append(t.data[i])
        return Tensor(out, (n, len(tensors)))
    raise NotImplementedError("stack axis")
