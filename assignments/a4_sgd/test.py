import sys
sys.path.append("../..")
from solution import solve
from optalg.interface.nlp_stochastic_traced import NLPStochasticTraced
import unittest
import numpy as np
from optalg.example_nlps.quadratic_stochastic import Quadratic_stochastic
from optalg.example_nlps.linear_least_squares_stochastic import Linear_least_squares_stochastic
from optalg.utils.timeout import run_with_timeout


def _solve(nlp):
    return run_with_timeout(lambda: solve(nlp))


tolerance = 5 * 1e-2


class testSolver(unittest.TestCase):
    """
    """

    def testLinearLS(self):

        A = np.array([[0.85880983, 0.42075752, 0.14625862],
                      [0.57705246, 0.76635021, 0.64077446],
                      [0.85916002, 0.86186594, 0.81231712],
                      [1., 0., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.]])

        b = np.zeros(6)

        problem = NLPStochasticTraced(Linear_least_squares_stochastic(A, b))

        x = _solve(problem)

        self.assertTrue(np.linalg.norm(problem.nlp.opt - x) < tolerance)

    def testQuadratic(self):

        Qs = [np.diag([1., 1, 1]), np.diag([2., 1, 4]), np.diag(
            [10., 0, 4]), np.diag([0., 1, 1]), np.diag([0., 0, 0.1])]

        gs = [np.array([0.13359639,
                        0.20692249,
                        0.4099303]),
              np.array([0.86686032,
                        0.26361185,
                        0.48886114]),
              np.array([0.01843443,
                        0.89926551,
                        0.3261251]),
              np.array([0.86751661,
                        0.95695649,
                        0.68315632]),
              np.array([0.90034537,
                        0.96245117,
                        0.84465686])]

        problem = NLPStochasticTraced(Quadratic_stochastic(Qs, gs))
        x = _solve(problem)

        self.assertTrue(np.linalg.norm(problem.nlp.opt - x) < tolerance)


# Run tests with:
# python3 test.py
# Too see help and options
# python3 test.py --help
if __name__ == "__main__":
    unittest.main()
