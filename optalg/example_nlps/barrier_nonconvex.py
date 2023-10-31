import math
import numpy as np

from ..interface.nlp import NLP
from ..interface.objective_type import OT


class Barrier_nonconvex(NLP):
    """
    f =  - || x - ones || ^2   - k sum( log(x_i)  )
    x in R^n
    k in R
    sos = []
    eq = []
    ineq = []
    bounds = ( [ -inf , -inf, ... ], [ inf, inf, ...] )
    x0 = .5 * ones
    """

    def __init__(self, n=2, k=1e-2):
        """
        """
        self.k = k
        self.n = n
        self.inf = 1e20
        self.delta = 1e-6

    def evaluate(self, x):
        """
        See Also
        ------
        NLP.evaluate
        """

        if np.sum(x < self.delta):
            return np.array([self.inf]), np.zeros((1, self.n))

        y = np.sum(x) - self.k * np.sum(np.log(x))
        J = np.ones(self.n) - self.k / x

        ones = np.ones(self.n)
        y -= np.dot(x - ones, x - ones)
        J -= 2 * (x - ones)

        return np.array([y]), J.reshape(1, -1)

    def getDimension(self):
        """
        See Also
        ------
        NLP.getDimension
        """
        return self.n

    def getFHessian(self, x):
        """
        See Also
        ------
        NLP.getFHessian
        """
        # -k log( x)
        # d/dx is -k / x
        # d^2/dx^2 is k / x^2
        H = np.diag(self.k / x ** 2)

        H += -2 * np.eye(self.n)
        return H

    def getFeatureTypes(self):
        """
        See Also
        ------
        NLP.getFeatureTypes
        """
        return [OT.f]

    def getInitializationSample(self):
        """
        See Also
        ------
        NLP.getInitializationSample
        """
        out = .5 * np.ones(self.n)
        return out

    def report(self, verbose):
        """
        See Also
        ------
        NLP.report
        """
        strOut = "Barrier Function"
        return strOut
