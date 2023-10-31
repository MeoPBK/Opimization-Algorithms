import numpy as np

try:
    from ..interface.nlp import NLP
    from ..interface.objective_type import OT
except BaseException:
    from interface.nlp import NLP
    from interface.objective_type import OT


class F_R_Eq(NLP):
    """
    .5 * xQx + || Rx - d ||
    s.t  Ax = b

    with Q=Q.T
    """

    def __init__(self, Q, R, d, A, b):
        self.Q = Q
        self.R = R
        self.d = d
        self.A = A
        self.b = b

    def evaluate(self, x):
        """
        See Also
        ------
        NLP.evaluate
        """

        # cost
        f = .5 * x @ self.Q @ x
        Jf = self.Q @ x
        r = self.R @ x - self.d
        Jr = self.R
        eq = self.A @ x - self.b
        Jeq = self.A

        return np.concatenate(([f], r, eq)), np.vstack((Jf, Jr, Jeq))

    def getDimension(self):
        """
        See Also
        ------
        NLP.getDimension
        """
        return self.Q.shape[0]

    def getFeatureTypes(self):
        """
        See Also
        ------
        NLP.getFeatureTypes
        """
        return [OT.f] + self.R.shape[0] * [OT.r] + self.A.shape[0] * [OT.eq]

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
        return np.copy(self.Q)

    def report(self, verbose):
        """
        See Also
        ------
        NLP.report
        """
        strOut = "f_r"
        return strOut
