import sys
sys.path.append("../..")
import numpy as np
from optalg.example_nlps.f_r import F_R as Problem
from optalg.interface.nlp import NLP
from optalg.utils.finite_diff import *
import numpy as np
import unittest


class test_f_r(unittest.TestCase):
    """
    test class Constrained0
    """
    problem = Problem

    def testJacobian(self):

        Q = np.array([[1., 0.], [0., 2.]])
        R = np.ones((2, 2))
        d = np.zeros(2)

        problem = self.problem(Q, R, d)

        x = np.array([0.1, 0.3])
        flag, _, _ = check_nlp(
            problem.evaluate, x, 1e-5, True)
        self.assertTrue(flag)

    def testHessian(self):

        Q = np.array([[1., 0.], [0., 2.]])
        R = np.ones((2, 2))
        d = np.zeros(2)

        problem = self.problem(Q, R, d)
        x = np.array([.1, .2])
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
