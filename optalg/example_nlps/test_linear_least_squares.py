import sys
import unittest
import numpy as np
import sys

sys.path.append("../..")
from optalg.utils.finite_diff import *
from optalg.interface.nlp import NLP
from optalg.example_nlps.linear_least_squares import LinearLeastSquares as Problem


class testProblem(unittest.TestCase):
    """
    test mathematical program Rosenbrock
    """

    problem = Problem

    def generateProblemA(self):
        A = np.random.rand(5, 5)
        b = np.random.rand(5)
        return self.problem(A, b)

    def testjacobianA(self):
        """
        """
        problem2 = self.generateProblemA()
        x = np.array([.1, .2, .3, .4, .5])
        flag, _, _ = check_nlp(
            problem2.evaluate, x, 1e-5)
        self.assertTrue(flag)


if __name__ == "__main__":
    unittest.main()
