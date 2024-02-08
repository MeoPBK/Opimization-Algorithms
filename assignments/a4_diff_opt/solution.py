import numpy as np
import sys
sys.path.append("../..")
from optalg.interface.nlp import NLP
import math


QP_0 = {
    "A": np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]),
    "b": np.array([1., 1., 1., 1.]),
    "x": np.array([.5, .5]),
    "yopt": np.array([.5, .5]),
    "lopt": np.array([0., 0., 0., 0.])
}


QP_1 = {
    "A": np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]),
    "b": np.array([1., 1., 1., 1.]),
    "x": np.array([2, 0]),
    "yopt": np.array([1, 0]),
    "lopt": np.array([2, 0., 0., 0.])
}

QP_2 = {
    "A": np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]),
    "b": np.array([1., 1., 1., 1.]),
    "x": np.array([2, 2]),
    "yopt": np.array([1, 1]),
    "lopt": np.array([2., 2., 0, 0]),


}

QPs = [QP_0, QP_1, QP_2]


def oracle(x: np.ndarray, A: np.ndarray, b: np.ndarray):
    """
    Oracle that provides solution to QP
    min_y || y - x ||^2
    s.t. Ay <= b

    Arguments:
    ---
    x: 1-D np.array
    A: 2-D np.array
    b: 1-D np.array


    Returns:
    yopt: 1-D np.array
    lopt: 1-D np.array

    """

    found = False
    num_qps = len(QPs)
    id = 0
    while (id < num_qps and not found):
        QP = QPs[id]
        if QP["A"].shape == A.shape and np.allclose(
                QP["A"],
                A) and np.allclose(
                QP["b"],
                b) and np.allclose(
                QP["x"],
                x):
            found = True
            yopt = QP["yopt"]
            lopt = QP["lopt"]
            # Check that the solution of the oracle is correct.

            print("hello")
            print(QP["A"] @ yopt - QP["b"])
            print(QP["A"] @ yopt - QP["b"] > 0)

            assert np.sum(QP["A"] @ yopt - QP["b"] > 0) == 0
            assert np.sum(lopt < 0) == 0
            assert np.sum(lopt * (QP["A"] @ yopt - QP["b"]) != 0) == 0
            assert np.linalg.norm(
                2 * (yopt - QP["x"]) + lopt @ QP["A"]) < 1e-12
        id += 1
    if not found:
        raise RuntimeError("QP is not in the database")

    return yopt, lopt


class DiffOpt(NLP):
    """
    min_x c^T yopt(x)

    where yopt(x) is the solution of min_y || y - x ||^2 s.t. Ay <= b

    x in R^2, y in R^2,  A in R^(2x2), and b in R^2
    """

    def __init__(self, c: np.ndarray, A: np.ndarray, b: np.ndarray):
        """
        Arguments
        ----
        c: 1-D np.array
        A: 2-D np.array
        b: 1-D
        """
        self.c = c
        self.A = A
        self.b = b

    def evaluate(self, x):
        """
        Returns the features (y) and the Jacobian (J) of the nonlinear program.
        In this case, we have 1 cost function.

        Therefore, the output should be:
            y: the feature (1-D np.ndarray of shape (1,))
            J: the Jacobian (2-D np.ndarray of shape (1,n))


        Notes:
        ---

        For an input x, you can get the optimal yopt calling the oracle:

        yopt, lopt = oracle(x, self.A, self.b)

        yopt is the optimal y.
        lopt is the value of the Lagrange multipliers.

        The Jacobian dyopt/dx has to be computed using the implicit function theorem on the KKT conditions
        of the optimization problem,

        min_y || y - x ||^2 s.t. Ay <= b

        where y is the variable and x is the parameter.

        """

        # y = ...
        # J = ...


        return y, J

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        # return the input dimensionality of the problem (size of x)
        return 2
