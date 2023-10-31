import numpy as np

from ..interface.nlp import NLP


class Quadratic (NLP):
    """
    x in R^n, A symmetric
    f =  .5 x^T A x
    sos = []
    eq = []
    ineq = []
    bounds = ( [ -inf , -inf], [ inf, inf] )
    """

    def __init__(self, A):
        self.A = A

    def evaluate(self, x):
        """
        """
        return np.array([.5 * np.dot(x, self.A @ x)]
                        ), (self.A @ x).reshape(1, -1)

    def getDimension(self):
        """
        """
        return self.A.shape[0]

    def getFHessian(self, x):
        """
        """
        return self.A

    def getInitializationSample(self):
        """
        """
        return np.ones(self.getDimension())

    def report(self, verbose):
        """
        """
        strOut = "Quadratic function, xAx"
        return strOut
