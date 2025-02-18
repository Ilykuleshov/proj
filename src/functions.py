import numpy as np
import numpy.linalg as la
from enum import Enum, auto
from sklearn.datasets import load_svmlight_file


class LossType(Enum):
    QUADRATIC = auto()
    LOGREG = auto()

class UtilityFunctions:
    def __init__(self, n, scale, task_params):
        self.scale = scale
        self.n = n

    def phi(self, x):
        raise NotImplementedError()

    def get_mu(self):
        raise NotImplementedError()

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

class Quadratic(UtilityFunctions):
    def __init__(self, n, scale, task_params):
        super().__init__(n, scale, task_params)
        self.kappa = task_params["kappa"]
        self.matrix = self.generate_matrix()

    def generate_matrix(self):
        des = np.random.uniform(low=1, high=self.kappa, size=self.n)
        des = 1 + (self.kappa - 1) * (des - min(des)) / (max(des) - min(des))
        s = np.diag(des)
        q, _ = la.qr(np.random.rand(self.n, self.n))

        return np.array([q.T @ s @ q]).squeeze()

    def get_mu(self):
        return min(la.eig(self.matrix)[0])

    def phi(self, x):
        return x.transpose() @ self.matrix @ x

class Logreg(UtilityFunctions):
    def __init__(self, n, scale, task_params):
        super().__init__(n, scale, task_params)
        data = load_svmlight_file("../datasets/mushrooms.txt")
        self.X, self.y = data[0].toarray(), data[1]
        print(self.X.shape)

    def logreg_loss(self, w):
        assert len(self.y) == self.X.shape[0]
        assert len(w) == self.X.shape[1]
        l = np.log(1 + np.exp(-self.X.dot(w) * self.y))
        m = self.y.shape[0]
        return np.sum(l) / m

    def phi(self, x):
        return self.logreg_loss(x)

    def get_mu(self):
        return 1