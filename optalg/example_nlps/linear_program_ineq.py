import numpy as np

try:
    from ..interface.nlp import NLP
    from ..interface.objective_type import OT
except BaseException:
    from interface.nlp import NLP
    from interface.objective_type import OT


class LinearProgramIneq(NLP):
    """
    x in R^n
    min sum 2 * x_i s.t. x_i >= 0
    """

    def __init__(self, n):
        self.n = n
        super().__init__()

    def evaluate(self, x):
        """
        See Also
        ------
        NLP.evaluate
        """

        # cost
        f = 2 * np.sum(x)
        grad = 2 * np.ones(self.getDimension())
        g = -x
        J = -1 * np.identity(self.getDimension())

        # constraint
        return np.concatenate((np.array([f]), g)), np.vstack((grad, J))

    def getDimension(self):
        """
        See Also
        ------
        NLP.getDimension
        """
        return self.n

    def getFeatureTypes(self):
        """
        See Also
        ------
        NLP.getFeatureTypes
        """
        return [OT.f] + self.getDimension() * [OT.ineq]

    def getInitializationSample(self):
        """
        See Also
        ------
        NLP.getInitializationSample
        """
        return np.ones(self.getDimension())

    def getFHessian(self, x):
        """
        Ref: https://www.wolframalpha.com/input/?i=hessian+of+++%28+a+-+x+%29+%5E+2+%2B+b+%28+y+-+x%5E2+%29+%5E+2

        See Also
        ------
        NLP.getFHessian
        """
        n = self.getDimension()
        hess = np.zeros((n, n))

        return hess

    def report(self, verbose):
        """
        See Also
        ------
        NLP.report
        """
        strOut = "linear program inequalities"
        return strOut
