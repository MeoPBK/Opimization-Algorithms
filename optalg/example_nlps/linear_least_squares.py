import numpy as np
import math

try:
    from ..optalg.interface.nlp import NLP
    from ..optalg.interface.objective_type import OT

except BaseException:
    from optalg.interface.nlp import NLP
    from optalg.interface.objective_type import OT


class LinearLeastSquares(NLP):
    """
    """

    def __init__(self, A, b, add_reg=True):
        """
        Arguments
        ----
        A: 2-D np.array
        b: 1-D np.array


        Linear least squares: phi(x) = [Ax-b , x]

        """
        self.A = A
        self.b = b
        self.add_reg = add_reg

    def evaluate(self, x):
        """
        See also:
        ----
        MathematicalProgram.evaluate
        """
        n = self.getDimension()
        y = self.A @ x - self.b
        yref = x

        J = self.A
        Jref = np.identity(n)

        if self.add_reg:
            return np.concatenate(([0], y, yref)), np.vstack(
                (np.zeros(self.getDimension()), J, Jref))

        else:
            return np.concatenate(([0], y)), np.vstack(
                (np.zeros(self.getDimension()), J))

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        return self.A.shape[1]

    def getFeatureTypes(self):
        if self.add_reg:
            return [OT.f] + (self.A.shape[0] + self.getDimension()) * [OT.r]
        else:
            return [OT.f] + self.A.shape[0] * [OT.r]

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return np.ones(self.getDimension())
