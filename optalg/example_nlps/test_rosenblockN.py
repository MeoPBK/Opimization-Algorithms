import sys
import unittest
import numpy as np
import sys

sys.path.append("..")

import sys
sys.path.append("../..")

from optalg.example_nlps.rosenbrockN import RosenbrockN as Problem
from optalg.interface.nlp import NLP
from optalg.utils.finite_diff import *
import numpy as np
import unittest
import math


class testProblem(unittest.TestCase):
    """
    test mathematical program Rosenbrock
    """

    problem = Problem

    def testValueAtOpt(self):
        """
        """

        problem2 = self.problem(2)
        phi, J = problem2.evaluate(np.ones(2))
        self.assertAlmostEqual(phi[0], 0)

        problem5 = self.problem(5)
        phi, J = problem5.evaluate(np.ones(5))
        self.assertAlmostEqual(phi[0], 0)

    def testjacobian(self):
        """
        """

        problem2 = self.problem(2)
        flag, _, _ = check_nlp(
            problem2.evaluate, np.array([.8, -.5]), 1e-5)
        self.assertTrue(flag)

        problem5 = self.problem(5)
        flag, _, _ = check_nlp(
            problem5.evaluate, np.array([.8, -.5, 0, 1, 0]), 1e-5)
        self.assertTrue(flag)

    def testoptimal(self):
        """
        """

        problem2 = self.problem(2)
        self.assertTrue(np.abs(problem2.evaluate(
            np.array([1, 1]))[0][0]) <= 1e-5)
        self.assertTrue(problem2.evaluate(np.array([1, 1]))[
                        0][0] <= problem2.evaluate(np.array([1, 1 + .001]))[0][0])
        self.assertTrue(problem2.evaluate(np.array([1, 1]))[
                        0][0] <= problem2.evaluate(np.array([1 + 0.001, 1]))[0][0])

        problem5 = self.problem(5)
        self.assertTrue(np.abs(problem5.evaluate(np.ones(5))[0][0]) <= 1e-5)
        for i in range(5):
            e = np.zeros(5)
            e[i] += .001
            self.assertTrue(problem5.evaluate(np.ones(5))[
                            0][0] <= problem5.evaluate(e)[0][0])

    def testhessian(self):
        """
        """

        problem2 = self.problem(2)
        x = np.array([.8, -.5])

        H = problem2.getFHessian(x)

        def f(x):
            return problem2.evaluate(x)[0][0]

        tol = 1e-4
        Hdiff = finite_diff_hess(f, x, tol)
        flag = np.allclose(H, Hdiff, 10 * tol, 10 * tol)
        self.assertTrue(flag)

        problem5 = self.problem(5)
        x = np.array([.8, -.5, -1, .1, .3])

        H = problem5.getFHessian(x)

        def f(x):
            return problem5.evaluate(x)[0][0]

        tol = 1e-4
        Hdiff = finite_diff_hess(f, x, tol)
        flag = np.allclose(H, Hdiff, 10 * tol, 10 * tol)
        self.assertTrue(flag)


if __name__ == "__main__":
    unittest.main()
