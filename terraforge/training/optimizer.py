"""Optimizers: SGD and Adam (list-backed)."""
import math


class SGD:
    def __init__(self, lr=3e-4, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}

    def step(self, params):
        for name, (value, grad) in params.items():
            v = self.velocities.get(name, 0.0)
            v = self.momentum * v + grad
            self.velocities[name] = v
            value -= self.lr * v


class Adam:
    def __init__(self, lr=3e-4, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params):
        self.t += 1
        b1, b2 = self.betas
        for name, (value, grad) in params.items():
            m = self.m.get(name, 0.0)
            v = self.v.get(name, 0.0)
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad * grad
            self.m[name] = m
            self.v[name] = v
            m_hat = m / (1 - b1 ** self.t)
            v_hat = v / (1 - b2 ** self.t)
            value -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
