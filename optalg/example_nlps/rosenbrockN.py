import numpy as np

from ..interface.nlp import NLP
from ..interface.objective_type import OT


class RosenbrockN(NLP):

    """
    https://www.sfu.ca/~ssurjano/rosen.html
    """

    def __init__(self, N):
        """
        Arguments:
            N: integer
        """
        self.N = N
        super().__init__()

    def evaluate(self, x):
        """
        See Also
        ------
        MathematicalProgram.evaluate
        """

        assert(len(x) == self.N)
        c = 0
        J = np.zeros(self.N)
        for i in range(0, self.N - 1):
            c += 100 * (x[i + 1] - x[i]**2)**2 + (x[i] - 1)**2
            J[i + 1] += 100 * (x[i + 1] - x[i]**2) * 2
            J[i] += 100 * (x[i + 1] - x[i]**2) * 2 * \
                (-2 * x[i]) + 2 * (x[i] - 1)

        return np.array([c]), J.reshape(1, -1)

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        return self.N

    def getFeatureTypes(self):
        """
        See Also
        ------
        MathematicalProgram.getFeatureTypes
        """
        return [OT.f]

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return .1 * np.ones(self.N)

    def getFHessian(self, x):
        """
        See Also
        ------
        MathematicalProgram.getFHessian
        """
        n = self.getDimension()
        H = np.zeros((n, n))
        for i in range(0, self.N - 1):
            # J[i+1]+= 200 * ( x[i+1] - x[i]**2 )
            H[i + 1, i + 1] += 100 * 2
            H[i + 1, i] += 100 * 2 * (-2) * x[i]
            # J[i]+= -400* ( x[i+1]x[i] - x[i]**3 )  + 2 x[i] -2
            H[i, i] += -400 * (x[i + 1] - 3 * x[i]**2) + 2
            H[i, i + 1] += -400 * x[i]

        return H

    def report(self, verbose):
        """
        See Also
        ------
        MathematicalProgram.report
        """
        strOut = "Rosenbrock Function"
        return strOut
