import numpy as np
import math
from optalg.interface.nlp_stochastic import NLP_stochastic


class Linear_least_squares_stochastic(NLP_stochastic):
    """
    || Ax - b ||^2 / N

    in every query, we can evaluate a single row

     ( A[i] x - b[i] ) ^2

    """

    def __init__(self, A: np.ndarray, b: np.ndarray):

        self.A = A
        self.b = b
        self.n = A.shape[1]
        self.num_samples = A.shape[0]
        self.opt = np.linalg.lstsq(A, b, rcond=None)[0]

    def evaluate_i(self, x: np.ndarray, i: int):
        y = (self.A[i] @ x - self.b[i]) ** 2
        J = 2 * (self.A[i] @ x - self.b[i]) * self.A[i]
        return np.array([y]), J.reshape(1, -1)

    def evaluate(self, x: np.ndarray):
        y = (np.dot(self.A @ x - self.b, self.A @ x - self.b)) / self.num_samples
        J = (2 * (self.A @ x - self.b) @ self.A) / self.num_samples
        return np.array([y]), J.reshape(1, -1)

    def getDimension(self):
        return self.n

    def getNumSamples(self):
        return self.num_samples

    def getInitializationSample(self):
        return np.ones(self.getDimension())
