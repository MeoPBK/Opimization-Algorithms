import numpy as np
import math
from optalg.interface.nlp_stochastic import NLP_stochastic


class Quadratic_stochastic(NLP_stochastic):
    """

    sum_i  [ x Q_i x  + g_i x ] / N

    Q_i = Q_i.T and positive definite

    in every query, we can evaluate a single term

    x Q_i x  + g_i x


    """

    def __init__(self, Qs, gs):

        assert len(Qs) == len(gs)
        self.Qs = Qs
        self.gs = gs
        self.n = Qs[0].shape[0]
        self.num_samples = len(self.Qs)
        self.Qall = np.sum(Qs, axis=0)
        self.gall = np.sum(gs, axis=0)

        # Grad: 2 Qx  + g = 0
        self.opt = np.linalg.lstsq(2 * self.Qall, -self.gall, rcond=None)[0]

    def evaluate_i(self, x: np.ndarray, i: int):
        y = x @ self.Qs[i] @ x + self.gs[i] @ x
        J = 2 * self.Qs[i] @ x + self.gs[i]
        return np.array([y]), J.reshape(1, -1)

    def evaluate(self, x: np.ndarray):
        y = (x @ self.Qall @ x + self.gall @ x) / self.num_samples
        J = (2 * self.Qall @ x + self.gall) / self.num_samples
        return np.array([y]), J.reshape(1, -1)

    def getDimension(self):
        return self.n

    def getNumSamples(self):
        return self.num_samples

    def getInitializationSample(self):
        return np.ones(self.getDimension())
