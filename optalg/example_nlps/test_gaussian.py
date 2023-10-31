import sys
sys.path.append("../..")

from optalg.example_nlps.gaussian import Gaussian as Problem
from optalg.interface.nlp import NLP
from optalg.utils.finite_diff import *
import numpy as np
import unittest
import math


class testGaussian(unittest.TestCase):
    """
    test class Constrained0
    """
    problem = Problem

    def testJacobian(self):
        x0 = np.array([1.5, 2.5])
        D = np.array([[2, 0], [0, 3]])
        problem = self.problem(x0, D)
        x = np.array([0.1, 0.3])
        flag, _, _ = check_nlp(
            problem.evaluate, x, 1e-5, True)
        self.assertTrue(flag)

    def testJacobian2(self):
        x0 = np.array([1.5, 2.5])
        D = np.array([[2, 0], [0, 3]])
        problem = self.problem(x0, D)
        x = np.array([0.5, 0.8])
        flag, _, _ = check_nlp(
            problem.evaluate, x, 1e-5, True)
        self.assertTrue(flag)

    def testJacobian3(self):
        x0 = np.array([1.5, 2.5])
        D = np.array([[2, 0], [0, 3]])
        problem = self.problem(x0, D)
        x = np.array([-0.5, 0.8])
        flag, _, _ = check_nlp(
            problem.evaluate, x, 1e-5, True)
        self.assertTrue(flag)

    def testHessian(self):

        x0 = np.array([1.5, 2.5])
        D = np.array([[2, 0], [0, 3]])
        problem = self.problem(x0, D)
        x = np.array([.5, .8])
        H = problem.getFHessian(x)

        def f(x):
            return problem.evaluate(x)[0][0]

        tol = 1e-4
        Hdiff = finite_diff_hess(f, x, tol)
        self.assertTrue(np.allclose(H, Hdiff, 10 * tol, 10 * tol))


# usage:
# print results in terminal
# python3 test.py
# store results in file
# python3 test.py out.log
if __name__ == "__main__":
    if len(sys.argv) == 2:
        log_file = sys.argv.pop()
        with open(log_file, "w") as f:
            runner = unittest.TextTestRunner(f, verbosity=2)
            unittest.main(testRunner=runner)
    else:
        unittest.main()
