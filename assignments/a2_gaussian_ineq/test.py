import sys  # noqa
sys.path.append("../..")  # noqa

import numpy as np
import unittest
from solution import NLP_Gaussina_ineq
from optalg.utils.finite_diff import finite_diff_J, finite_diff_hess


# import the test classes
class test_NLP_Gaussian_ineq(unittest.TestCase):
    """
    test NLP_Gaussina_ineq
    """

    def testValue(self):

        D = np.eye(2)
        A = np.ones((3, 2))
        b = np.array([2., 2., 2.])
        x0 = np.array([1., .1])

        x0 = np.ones(2)
        problem = NLP_Gaussina_ineq(x0, D, A, b)
        x = np.ones(2)
        y, _ = problem.evaluate(x)
        solution = np.array([-1., 0, 0, 0])
        self.assertTrue(np.allclose(y, solution, 1e-4))

    def testValue2(self):

        D = np.eye(2)
        A = np.ones((3, 2))
        b = np.array([2., 2., 2.])
        x0 = np.array([1., .1])

        x0 = np.ones(2)
        problem = NLP_Gaussina_ineq(x0, D, A, b)
        x = np.zeros(2)
        y, _ = problem.evaluate(x)
        solution = np.array([- np.exp(-2), -2., -2., -2.])
        self.assertTrue(np.allclose(y, solution, 1e-4))

    def testJacobian(self):

        D = np.array([[2, 1], [1, 2]])
        A = np.array([[1., 2.], [3., 4.]])
        b = np.array([0., 1.])
        x0 = np.array([.5, .1])

        problem = NLP_Gaussina_ineq(x0, D, A, b)
        x = np.array([-1, .5])
        y, J = problem.evaluate(x)
        eps = 1e-5
        Jdiff = finite_diff_J(problem, x, eps)
        self.assertTrue(np.allclose(J, Jdiff, eps * 10))

    def testHessian(self):

        D = np.array([[2, 1], [1, 2]])
        A = np.array([[1., 2.], [3., 4.]])
        b = np.array([0., 1.])
        x0 = np.array([.5, .1])

        problem = NLP_Gaussina_ineq(x0, D, A, b)

        x = np.array([-1, .1])
        H = problem.getFHessian(x)

        def f(x):
            return problem.evaluate(x)[0][0]

        tol = 1e-4
        Hdiff = finite_diff_hess(f, x, tol)
        self.assertTrue(np.allclose(H, Hdiff, 10 * tol, 10 * tol))


if __name__ == "__main__":
    unittest.main()
