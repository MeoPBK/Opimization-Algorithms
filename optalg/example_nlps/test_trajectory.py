import sys
import unittest
import numpy as np
import sys

sys.path.append("../..")
from optalg.utils.finite_diff import *
from optalg.example_nlps.trajectory import PointTrajOpt as Problem


class testProblem(unittest.TestCase):
    """
    """

    problem = Problem

    def generateProblemA(self):
        N = 4
        return self.problem(N)

    # def generateProblemB(self):
    #     N = 3
    #     q0 = np.array([-.1,0,-.4])
    #     pr = np.array([-2., 0.5])
    #     l = 0.1
    #     return  self.problem( N, q0, pr, l)

    def testjacobianA(self):
        """
        """
        problem2 = self.generateProblemA()
        x = np.arange(problem2.getDimension())
        # x = np.ones(10)
        flag, _, _ = check_nlp(
            problem2.evaluate, x, 1e-5, False)
        self.assertTrue(flag)

    def testhessian(self):
        """
        """

        problem2 = self.generateProblemA()
        x = np.random.rand(problem2.getDimension())
        # x = np.arange(10)

        H = problem2.getFHessian(x)

        def f(x):
            return problem2.evaluate(x)[0][0]

        tol = 1e-4
        Hdiff = finite_diff_hess(f, x, tol)
        print("Hs")
        print(H)
        print(Hdiff)
        print(H - Hdiff)
        flag = np.allclose(H, Hdiff, 10 * tol, 10 * tol)
        self.assertTrue(flag)

    # def testjacobianB(self):
    #     """
    #     """
    #     problem2 = self.generateProblemB()
    #     x = np.array([-.1,.0,.3])
    #     flag, _, _ = check_mathematical_program(
    #         problem2.evaluate, x , 1e-5)
    #     self.assertTrue(flag)

    # HOW TO GENERATE THE PROBLEMS?
    # HOW TO choose q0 and pr?


if __name__ == "__main__":
    unittest.main()
