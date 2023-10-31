import sys
import numpy as np

try:
    from ..interface.nlp import NLP
    from ..interface.objective_type import OT
except BaseException:
    from interface.nlp import NLP
    from interface.objective_type import OT


class NonlinearA(NLP):

    """
    REF: https://github.com/ethz-adrl/ifopt
    + term || x  - (1,0)  ||
    solution is:
    x∗=(1,0)
    """

    def __init__(self):
        self.ref0 = 1.
        self.ref1 = 0.
        self.w = 10.

    def evaluate(self, x):
        """
        See Also
        ------
        NLP.evaluate
        """

        # cost
        f = -(x[1] - 2)**2 + .5 * self.w * (x[0] - self.ref0) ** 2 + .5 * \
            self.w * (x[1] - self.ref1) ** 2
        grad_f = np.array([0, -2 * (x[1] - 2)])
        grad_f += self.w * np.array([x[0] - self.ref0, x[1] - self.ref1])

        # constraint
        h = x[0]**2 + x[1] - 1
        dh = np.array([2 * x[0], 1])

        # bounds (as normal constraint)
        bU = x[0] - 1
        bL = -1 - x[0]

        Jb = np.zeros((2, 2))
        Jb[0, 0] = 1
        Jb[1, 0] = -1

        return np.array([f, h, bU, bL]), np.vstack((grad_f, dh, Jb))

    def getDimension(self):
        """
        See Also
        ------
        NLP.getDimension
        """
        return 2

    def getFeatureTypes(self):
        """
        See Also
        ------
        NLP.getFeatureTypes
        """
        return [OT.f, OT.eq, OT.ineq, OT.ineq]

    def getInitializationSample(self):
        """
        See Also
        ------
        NLP.getInitializationSample
        """
        return np.array([3.5, 1.5])

    def getFHessian(self, x):
        """
        Ref: https://www.wolframalpha.com/input/?i=hessian+of+++%28+a+-+x+%29+%5E+2+%2B+b+%28+y+-+x%5E2+%29+%5E+2

        See Also
        ------
        NLP.getFHessian
        """
        hess = np.zeros((2, 2))

        # we fill first lower triangular
        hess[1, 1] = -2
        hess[0, 0] += self.w * 1.
        hess[1, 1] += self.w * 1.

        return hess

    def report(self, verbose):
        """
        See Also
        ------
        NLP.report
        """
        strOut = "nonlinearA"
        return strOut
