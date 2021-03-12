import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize
from tqdm import trange
import pickle
import os


from functions import UtilityFunctions


try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


class Experiment:
    def __init__(self,
                 n: int,
                 kappa: float,
                 scale: float,
                 alpha: float,
                 beta: float,
                 tau: float = None,
                 gamma: float = None,
                 batch_size: float = None,
                 save_folder: str = None,
                 ):

        self.n = n
        self.kappa = kappa
        self.scale = scale
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.save_folder = save_folder

        if SummaryWriter is not None and self.save_folder is not None:
            self.writer = SummaryWriter(save_folder)

        self.matrix = self.generate_matrix()
        self.mu = min(la.eig(self.matrix)[0])
        self.starting_point = np.ones(n) * 10
        self.functions = UtilityFunctions(scale, n, self.matrix)

    def generate_matrix(self):
        des = np.random.uniform(low=1, high=self.kappa, size=self.n)
        des = 1 + (self.kappa - 1) * (des - min(des)) / (max(des) - min(des))
        s = np.diag(des)
        q, _ = la.qr(np.random.rand(self.n, self.n))

        return np.array([q.T @ s @ q]).squeeze()

    def find_optimal_solution(self):
        z = minimize(self.functions.phi, self.starting_point)
        return z

    def get_gamma(self, iteration):
        if self.gamma is not None:
            return self.gamma

        return 1 / (self.mu * (iteration + 1))

    def get_tau(self, iteration):
        if self.tau is not None:
            return self.tau
        return 1 / ((iteration + 1) ** self.beta)

    def get_batch_size(self, iteration):
        if self.batch_size is not None:
            return self.batch_size

        return int((iteration + 1) ** self.alpha)

    def run(self, num_iters):
        errors = []
        xs = []

        opt = self.find_optimal_solution()

        loop = trange(1000, num_iters)
        x_prev = self.starting_point
        for k in loop:

            tau = self.get_tau(k)
            gamma = self.get_gamma(k)
            batch_size = self.get_batch_size(k)

            x_next = self.functions.step(x_prev, gamma, tau, batch_size)
            err = la.norm(x_next - opt.x, 2)
            loop.set_description("error: %.3e; bsz: %d" % (err, batch_size))
            errors.append(err)
            self.writer.add_scalar('error', err / errors[0], k)
            xs.append(x_next)
            x_prev = x_next

        data = {"x": xs, "errors": errors, "matrix": self.matrix}
        with open(os.path.join(self.save_folder, "data.pkl"), "wb") as file:
            pickle.dump(data, file)

        print("Optimizing finished. Final error: %.3e, f(x) = %.3e, optimal = %.3e"
              % (err, self.functions.phi(x_next), opt.fun))

