import sys
sys.path.append("../..")
from solution import NLP_nonlinear
from optalg.interface.nlp import NLP
from optalg.interface.nlp_timeout import NLPTimeout
from optalg.utils.finite_diff import finite_diff_J, finite_diff_hess
from optalg.utils.timeout import run_with_timeout
import numpy as np
import unittest


# import the test classes

class test_NLP_nonlinear(unittest.TestCase):
    """
    test NLP_nonlinear
    """
    #

    def testValue(self):

        C = np.ones((2, 2))
        problem = NLPTimeout(NLP_nonlinear(C))
        x = np.ones(2)
        y, _ = problem.evaluate(x)
        value = y[0]
        solution = 1. / np.sqrt(8)
        self.assertAlmostEqual(value, solution)

    def testJacobian(self):

        C = np.ones((2, 2))
        problem = NLPTimeout(NLP_nonlinear(C))
        x = np.array([-1, .5])
        _, J = problem.evaluate(x)
        eps = 1e-5
        Jdiff = run_with_timeout(lambda: finite_diff_J(problem, x, eps))
        self.assertTrue(np.allclose(J, Jdiff, eps * 10))

    def testHessian(self):

        C = np.ones((2, 2))
        problem = NLPTimeout(NLP_nonlinear(C))
        x = np.array([-1.1, .1])
        H = problem.getFHessian(x)

        def f(x):
            return problem.evaluate(x)[0][0]

        tol = 1e-4
        Hdiff = run_with_timeout(lambda: finite_diff_hess(f, x, tol))
        self.assertTrue(np.allclose(H, Hdiff, 10 * tol, 10 * tol))


# usage:
# python3 test.py

if __name__ == "__main__":
    unittest.main()
