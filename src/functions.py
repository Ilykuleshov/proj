import numpy as np
import numpy.linalg as la


class UtilityFunctions:
    def __init__(self, scale, n, matrix):
        self.scale = scale
        self.n = n
        self.matrix = matrix

    def phi(self, x):
        return x.transpose() @ self.matrix @ x

    def noisy_phi(self, x):
        xi = np.random.normal(0, self.scale)
        return self.phi(x) + xi

    def grad(self, x, tau, e):
        return self.n * (self.noisy_phi(x + tau * e) - self.noisy_phi(x - tau * e)) / (2 * tau) * e

    def step(self, x_prev, gamma, tau, batch_size):
        g = np.zeros(self.n)
        for _ in range(batch_size):
            e = np.random.normal(0, 1, self.n)
            e /= la.norm(e, 2)
            g += self.grad(x_prev, tau, e)

        g /= batch_size
        return x_prev - gamma * g
