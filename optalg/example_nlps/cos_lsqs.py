# gradient:
# sin(.5*|| Ax -b ||)(Ax-b) A
# hessian
# cos(.5 * || Ax -b ||) (Ax-b) A (Ax-b) A + sin(.5*|| Ax -b ||) A^T A

import math
import numpy as np

from ..interface.nlp import NLP
from ..interface.objective_type import OT


class Cos_lsqs(NLP):
    """
    f = - cos( .5 * || Ax - b ||^2 ) + alpha || x || ^ 2

    """

    def __init__(self, A=np.eye(2), b=np.zeros(2)):
        """
        """
        self.A = A
        self.b = b
        self.alpha = .1

    def evaluate(self, x):
        """
        See Also
        ------
        NLP.evaluate
        """

        A = self.A
        b = self.b
        r = A @ x - b
        Axb2 = np.dot(r, r)
        y = - np.cos(.5 * Axb2) + self.alpha * np.dot(x, x)
        J = np.sin(.5 * Axb2) * r @ A + 2 * self.alpha * x
        return np.array([y]), J.reshape(1, -1)

    def getDimension(self):
        """
        See Also
        ------
        NLP.getDimension
        """
        return self.A.shape[1]

    def getFHessian(self, x):
        """
        See Also
        ------
        NLP.getFHessian
        """
        A = self.A
        b = self.b
        r = A @ x - b
        Axb2 = np.dot(r, r)
        H1 = np.cos(.5 * Axb2) * np.outer(r@A, r@A)
        H2 = np.sin(.5 * Axb2) * A.T @ A

        return H1 + H2 + 2 * self.alpha * np.eye(len(x))

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
        n = self.A.shape[1]
        return .1 * np.ones(n)

    def report(self, verbose):
        """
        See Also
        ------
        NLP.report
        """
        strOut = "Barrier Function"
        return strOut
