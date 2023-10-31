import numpy as np
import math

try:
    from ..interface.nlp import NLP
    from ..interface.objective_type import OT
except BaseException:
    from interface.nlp import NLP
    from interface.objective_type import OT


try:
    from .logistic import Logistic
except BaseException:
    from logistic import Logistic


class LogisticWithBounds(NLP):

    """
    """

    def __init__(self):
        """
        """

        # self.K = 1.0
        # self.r = 10.0
        # self.t0 = .5

        # INIT SAMPLE = np.array([1, 1, 1])

        self.unconstrained = Logistic()
        self.UB = 2 * np.ones(3)
        self.LB = 0 * np.ones(3)

    def evaluate(self, x):
        """
        See Also
        ------
        NLP.evaluate
        """
        phi, j = self.unconstrained.evaluate(x)
        # x <= UB
        ub = x - self.UB
        # -x <= -LB
        lb = -x + self.LB
        return np.concatenate((phi, ub, lb)), np.vstack(
            (j, np.identity(3), -1 * np.identity(3)))

    def getDimension(self):
        """
        See Also
        ------
        NLP.getDimension
        """
        return self.unconstrained.getDimension()

    def getFeatureTypes(self):
        """
        See Also
        ------
        NLP.getFeatureTypes
        """
        return self.unconstrained.getFeatureTypes() + 6 * [OT.ineq]

    def getInitializationSample(self):
        """
        See Also
        ------
        NLP.getInitializationSample
        """
        return self.unconstrained.getInitializationSample()

    def report(self, verbose):
        """
        See Also
        ------
        NLP.report
        """
        strOut = "Logistic Regression"
        return strOut
