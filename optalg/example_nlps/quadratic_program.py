import sys
import numpy as np

try:
    from ..interface.nlp import NLP
    from ..interface.objective_type import OT
except BaseException:
    from interface.nlp import NLP
    from interface.objective_type import OT


class QuadraticProgram(NLP):

    def __init__(self, H, g, Aineq=[], bineq=[], Aeq=[], beq=[], lb=[], ub=[]):
        """
        NOTE: H must be symmetric!!

        min .5 * x.T * H * x + g.T * x
        s.t.
        Aineq * x <= b
        Aeq * x = b
        lb <= x <= ub
        """
        self.H = H
        self.g = g
        self.Aineq = Aineq
        self.bineq = bineq
        self.Aeq = Aeq
        self.beq = beq
        self.lb = lb
        self.ub = ub

    def evaluate(self, x):
        """
        See Also
        ------
        NLP.evaluate
        """

        # cost
        f = 0.5 * x @ (self.H @ x) + self.g @ x
        df = self.H @ x + self.g

        phi = np.array([f])
        J = df.reshape(1, -1)

        if len(self.Aineq) > 0:
            phi = np.concatenate((phi, self.Aineq @ x - self.bineq))
            J = np.vstack((J, self.Aineq))

        if len(self.Aeq) > 0:
            phi = np.concatenate((phi, self.Aeq @ x - self.beq))
            J = np.vstack((J, self.Aeq))

        return phi, J

    def getDimension(self):
        """
        See Also
        ------
        NLP.getDimension
        """
        return self.H.shape[0]

    def getFeatureTypes(self):
        """
        See Also
        ------
        NLP.getFeatureTypes
        """
        return [OT.f] + len(self.Aineq) * [OT.ineq] + len(self.Aeq) * [OT.eq]

    def getInitializationSample(self):
        """
        See Also
        ------
        NLP.getInitializationSample
        """
        return np.ones(self.getDimension())

    def getFHessian(self, x):
        """

        See Also
        ------
        NLP.getFHessian
        """
        return self.H

    def report(self, verbose):
        """
        See Also
        ------
        NLP.report
        """
        strOut = "Quadratic Program"
        return strOut
