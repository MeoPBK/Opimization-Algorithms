import sys
sys.path.append("../..")
from solution import solve
from optalg.example_nlps.hole import Hole
from optalg.example_nlps.barrier import Barrier
from optalg.example_nlps.quadratic_identity_2 import QuadraticIdentity2
from optalg.interface.nlp_traced import NLPTraced
from optalg.utils.timeout import run_with_timeout
import unittest
import numpy as np


class testGradientDescent(unittest.TestCase):
    """
    test Gradient Descent Solver
    """

    def testQuadraticIdentity(self):

        problem = NLPTraced(QuadraticIdentity2())

        x = solve(problem)
        fout = problem.evaluate(x)[0][0]
        solution = np.zeros(2)
        fopt = problem.evaluate(solution)[0][0]
        self.assertTrue(fout - fopt < 1e-3)

    def testHole(self):

        def make_C_exercise1(n, c):
            """
            n: integer
            c: float
            """
            C = np.zeros((n, n))
            for i in range(n):
                C[i, i] = c ** (float(i - 1) / (n - 1))
            return C

        C = make_C_exercise1(3, .1)
        problem = NLPTraced(Hole(C, 1.5))
        solution = np.zeros(3)
        x = solve(problem)
        fout = problem.evaluate(x)[0][0]
        fopt = problem.evaluate(solution)[0][0]
        self.assertTrue(fout - fopt < 1e-3)

    def testBarrier(self):

        problem = NLPTraced(Barrier())
        solution = 0.01 * np.ones(2)
        x = solve(problem)
        fout = problem.evaluate(x)[0][0]
        fopt = problem.evaluate(solution)[0][0]
        self.assertTrue(np.linalg.norm(fout - fopt) < 1e-3)


# Run tests with:
# python3 test.py
# Too see help and options
# python3 test.py --help
if __name__ == "__main__":
    unittest.main()
