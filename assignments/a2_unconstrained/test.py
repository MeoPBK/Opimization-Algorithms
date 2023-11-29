import sys  # noqa
sys.path.append("../..")   # noqa

import unittest
import numpy as np
from solution import solve
from optalg.example_nlps.hole import Hole
from optalg.example_nlps.barrier import Barrier
from optalg.example_nlps.rosenbrock import Rosenbrock
from optalg.example_nlps.rosenbrockN import RosenbrockN
from optalg.example_nlps.cos_lsqs import Cos_lsqs
from optalg.example_nlps.quadratic import Quadratic
from optalg.interface.nlp_traced import NLPTraced
from optalg.example_nlps.barrier_nonconvex import Barrier_nonconvex


# NOTE: This does not work in Windows!
# def _solve(nlp):
#     return run_with_timeout(lambda: solve(nlp))

def _solve(nlp):
    return solve(nlp)


TOLERANCE = 1e-3


class testSolverUnconstrained(unittest.TestCase):
    """
    """

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

        C = make_C_exercise1(3, .01)
        problem = NLPTraced(Hole(C, 1.5))
        solution = np.zeros(3)
        x = _solve(problem)
        fsol = problem.evaluate(solution)[0][0]
        fout = problem.evaluate(x)[0][0]
        self.assertTrue(fout < fsol + TOLERANCE)

    def testBarrier(self):

        problem = NLPTraced(Barrier())
        solution = 0.01 * np.ones(2)
        x = _solve(problem)
        fsol = problem.evaluate(solution)[0][0]
        fout = problem.evaluate(x)[0][0]
        self.assertTrue(fout < fsol + TOLERANCE)

    def testRosenbrock_easy(self):

        problem = NLPTraced(Rosenbrock(1., 1.5))

        problem.getInitializationSample = lambda: np.array([1.2, 1.2])

        x = _solve(problem)

        solution = np.array([1., 1.])

        fsol = problem.evaluate(solution)[0][0]
        fout = problem.evaluate(x)[0][0]
        self.assertTrue(fout < fsol + TOLERANCE)
        # self.assertTrue(np.linalg.norm(x - solution) < 1e-3)

    def testQuadratic(self):
        def make_C_exercise1(n, c):
            """
            n: integer
            c: float
            """
            C = np.zeros((n, n))
            for i in range(n):
                C[i, i] = c ** (float(i - 1) / (n - 1))
            return C

        problem = NLPTraced(
            Quadratic(make_C_exercise1(10, .00009)), max_evaluate=1000)

        x = _solve(problem)
        solution = np.zeros(problem.getDimension())

        fsol = problem.evaluate(solution)[0][0]
        fout = problem.evaluate(x)[0][0]
        self.assertTrue(fout < fsol + TOLERANCE)

    def testBarrier_large(self):

        n = 10
        problem = NLPTraced(Barrier(n=n))
        problem.getInitializationSample = lambda: np.ones(
            n) + 0.1 * np.arange(n)
        solution = 0.01 * np.ones(n)

        x = _solve(problem)
        fsol = problem.evaluate(solution)[0][0]
        fout = problem.evaluate(x)[0][0]
        self.assertTrue(fout < fsol + TOLERANCE)

        # self.assertTrue(np.linalg.norm(x - solution) < 1e-3)

    def testRosenbrock(self):
        problem = NLPTraced(Rosenbrock(2, 100))
        solution = np.array([2, 4])
        x = _solve(problem)
        fsol = problem.evaluate(solution)[0][0]
        fout = problem.evaluate(x)[0][0]
        self.assertTrue(fout < fsol + TOLERANCE)

    def testRosenbrockN(self):
        N = 7
        problem = NLPTraced(RosenbrockN(N))
        x = _solve(problem)
        solution = np.ones(N)
        fsol = problem.evaluate(solution)[0][0]
        fout = problem.evaluate(x)[0][0]
        self.assertTrue(fout < fsol + TOLERANCE)

    def testCosLsqs(self):
        A = .1 * np.array([[1., 2., 3.], [4, 5, 6]])
        b = np.zeros(2)
        problem = NLPTraced(Cos_lsqs(A, b))
        x = _solve(problem)
        solution = np.zeros(3)
        fsol = problem.evaluate(solution)[0][0]
        fout = problem.evaluate(x)[0][0]
        self.assertTrue(fout < fsol + TOLERANCE)

    def testRosenbrock_larger(self):
        N = 9
        problem = NLPTraced(RosenbrockN(N))
        x = _solve(problem)
        solution = np.ones(N)
        fsol = problem.evaluate(solution)[0][0]
        fout = problem.evaluate(x)[0][0]
        self.assertTrue(fout < fsol + TOLERANCE)

    def testBarrier_nonconvex(self):
        N = 4
        problem = NLPTraced(Barrier_nonconvex(N))
        x = _solve(problem)
        solution = 0.00334 * np.ones(N)
        fsol = problem.evaluate(solution)[0][0]
        fout = problem.evaluate(x)[0][0]
        self.assertTrue(fout < fsol + TOLERANCE)

    def testHole_large(self):

        def make_C_exercise1(n, c):
            """
            n: integer
            c: float
            """
            C = np.zeros((n, n))
            for i in range(n):
                C[i, i] = c ** (float(i - 1) / (n - 1))
            return C

        C = make_C_exercise1(6, .01)
        problem = NLPTraced(Hole(C, 1.5))
        solution = np.zeros(6)
        x = _solve(problem)
        fsol = problem.evaluate(solution)[0][0]
        fout = problem.evaluate(x)[0][0]
        self.assertTrue(fout < fsol + TOLERANCE)


if __name__ == "__main__":
    unittest.main()
