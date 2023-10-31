import math
import numpy as np

from optalg.example_nlps.hole import Hole
from optalg.interface.nlp import NLP
from optalg.interface.objective_type import OT


class HoleIneq(NLP):
    """
    f =  x^T C x  / ( a*a + x^T C x )
    s.t.
    Ax < b
    sos = []
    eq = []
    bounds = ( [ -inf , -inf, ... ], [ inf, inf, ...] )
    """

    def __init__(self, C, a, A, b):
        """
        C: np.array 2d
        a: float
        """
        self.hole = Hole(C, a)
        self.A = A
        self.b = b
        self.n = C.shape[0]

    def evaluate(self, x):
        """
        See Also
        ------
        NLP.evaluate
        """

        f, df = self.hole.evaluate(x)
        ineq = self.A @ x - self.b
        Jineq = self.A
        y = np.concatenate((f, ineq))
        J = np.vstack((df, Jineq))
        return y, J

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
        return self.hole.getFHessian(x)

    def getFeatureTypes(self):
        """
        See Also
        ------
        NLP.getFeatureTypes
        """
        return [OT.f] + self.A.shape[0] * [OT.ineq]

    def getInitializationSample(self):
        """
        See Also
        ------
        NLP.getInitializationSample
        """
        return np.ones(self.n)

    def report(self, verbose):
        """
        See Also
        ------
        NLP.report
        """
        strOut = "Hole function C"
        return strOut
