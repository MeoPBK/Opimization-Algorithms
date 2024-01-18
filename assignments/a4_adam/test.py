import sys
sys.path.append("../..")
from solution import solve
from optalg.interface.nlp_stochastic_traced import NLPStochasticTraced
import unittest
import numpy as np
from optalg.example_nlps.quadratic_stochastic import Quadratic_stochastic
from optalg.example_nlps.linear_least_squares_stochastic import Linear_least_squares_stochastic
from optalg.utils.timeout import run_with_timeout


# def _solve(nlp):
#     return solve(nlp)


def _solve(nlp):
    return run_with_timeout(lambda: solve(nlp))


tolerance = 5 * 1e-2


class testSolver(unittest.TestCase):
    """
    """

    def testLinear(self):

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

    def testLinear_2(self):

        A = np.array([[0.90657231, 0.59956825, 0.79181794],
                      [0.28089784, 0.27284522, 0.6512166],
                      [0.21775311, 0.81865644, 0.95784162],
                      [0.47410968, 0.98784787, 0.5146119],
                      [0.83881493, 0.60040309, 0.19827599],
                      [0.90834492, 0.66617969, 0.83357351],
                      [0.97028032, 0.67150341, 0.58111735],
                      [0.55876076, 0.36616212, 0.87166266],
                      [0.7202108, 0.85757553, 0.41819739],
                      [0.54088318, 0.46910575, 0.45215856]])

        b = np.zeros(10)

        problem = NLPStochasticTraced(Linear_least_squares_stochastic(A, b))

        x = _solve(problem)

        self.assertTrue(np.linalg.norm(problem.nlp.opt - x) < tolerance)

    def testLinear_3(self):

        A = np.array([[0.90657231, 0.59956825, 0.79181794],
                      [0.28089784, 0.27284522, 0.6512166],
                      [0.21775311, 0.81865644, 0.95784162],
                      [0.47410968, 0.98784787, 0.5146119],
                      [0.83881493, 0.60040309, 0.19827599],
                      [0.90834492, 0.66617969, 0.83357351],
                      [0.97028032, 0.67150341, 0.58111735],
                      [0.55876076, 0.36616212, 0.87166266],
                      [0.7202108, 0.85757553, 0.41819739],
                      [0.54088318, 0.46910575, 0.45215856]])

        b = np.array([0.08833689,
                      0.07526668,
                      0.01135477,
                      0.06107546,
                      0.05448763,
                      0.08905451,
                      0.06238092,
                      0.09981542,
                      0.0489913,
                      0.02405487])

        problem = NLPStochasticTraced(Linear_least_squares_stochastic(A, b))

        x = _solve(problem)

        self.assertTrue(np.linalg.norm(problem.nlp.opt - x) < tolerance)

    def testQuadratic(self):

        Qs = [np.diag([1, 1, 1]), np.diag([2, 1, 4]), np.diag(
            [10, 0, 4]), np.diag([0, 1, 1]), np.diag([0, 0, 0.1])]

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

    def testQuadratic_2(self):

        Qs = 5 * [np.diag([100., 1., 0.1])]

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
