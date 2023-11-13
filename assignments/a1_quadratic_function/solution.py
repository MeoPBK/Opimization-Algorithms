import sys
sys.path.append("../..")


from optalg.interface.nlp import NLP
import numpy as np


class NLP_xCCx(NLP):
    """
    Nonlinear program with quadratic cost  x^T C^T C x
    x in R^n
    C in R^(m x n)
    ^T means transpose
    feature types: [ OT.f ]

    """

    def __init__(self, C):
        """
        """
        self.C = C

    def evaluate(self, x):
        """
        Returns the features and the Jacobians
        of a nonlinear program.
        In this case, we have a single feature (the cost function)
        because there are no constraints or residual terms.
        Therefore, the output should be:
            y: the feature (1-D np.ndarray of shape (1,))
            J: the jacobian (2-D np.ndarray of shape (1,n))

        See also:
        ----
        NLP.evaluate
        """

        # Add code to compute the feature (cost) and the Jacobian
        y = (x.T @ self.C.T @ self.C @ x).reshape((1,))
        J = (2 * x.T @ self.C.T @ self.C).reshape((1, -1))


        return y, J

    def getDimension(self):
        """
        Return the dimensionality of the variable x

        See Also
        ------
        NLP.getDimension
        """

        # n =
        return self.C.size[1]

    def getFHessian(self, x):
        """
        Returns the hessian of the cost term.
        The output should be:
            H: the hessian (2-D np.ndarray of shape (n,n))

        See Also
        ------
        NLP.getFHessian
        """

        # Add code to compute the Hessian
        A = self.C.T @ self.C
        H = (A + A.T)


        return H

    def getInitializationSample(self):
        """
        See Also
        ------
        NLP.getInitializationSample
        """
        return np.ones(self.getDimension())

    def report(self, verbose):
        """
        See Also
        ------
        NLP.report
        """
        return "Quadratic function x^T C^T C x "
