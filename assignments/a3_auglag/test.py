import sys

sys.path.append("../..")
import unittest
import numpy as np
from solution import solve
from optalg.example_nlps.linear_program_ineq import LinearProgramIneq
from optalg.example_nlps.quadratic_program import QuadraticProgram
from optalg.example_nlps.halfcircle import HalfCircle
from optalg.example_nlps.logistic_bounds import LogisticWithBounds
from optalg.example_nlps.nonlinearA import NonlinearA
from optalg.interface.nlp_traced import NLPTraced
from optalg.example_nlps.f_r_eq import F_R_Eq
from optalg.interface.objective_type import OT
from optalg.example_nlps.quarter_circle import QuarterCircle
from optalg.example_nlps.halfcircle import HalfCircle2


FACTOR = 30
MAX_IT = 10000


def _solve(nlp):
    return solve(nlp)


def assertTrue(x):
    if not x:
        raise AssertionError()


def check_sol(problem, x, solution):
    fsol = problem.evaluate(solution)[0][0]
    fout = problem.evaluate(x)[0][0]

    id_r = [i for i, t in enumerate(problem.getFeatureTypes()) if t == OT.r]
    id_ineq = [i for i, t in enumerate(problem.getFeatureTypes()) if t == OT.ineq]
    id_eq = [i for i, t in enumerate(problem.getFeatureTypes()) if t == OT.eq]

    if len(id_r) > 0:
        r = problem.evaluate(x)[0][id_r]
        rsol = problem.evaluate(solution)[0][id_r]
        assertTrue(fout +  np.dot(r, r) < fsol +  np.dot(rsol, rsol) + 1e-3)

    else:
        assertTrue(fout < fsol + 1e-3)
    if len(id_eq) > 0:
        assertTrue(np.max(np.abs(problem.evaluate(x)[0][id_eq])) < 1e-2)
    if len(id_ineq) > 0:
        assertTrue(
            np.max( problem.evaluate(x)[0][id_ineq]) < 1e-2
        )


class testAuglag(unittest.TestCase):
    def testLinear1(self):
        problem = NLPTraced(LinearProgramIneq(2), max_evaluate=39 * FACTOR)
        x = _solve(problem)
        solution = np.zeros(2)
        check_sol(problem, x, solution)

    def testQuadraticIneq(self):
        """ """
        H = np.array([[1.0, -1.0], [-1.0, 2.0]])
        g = np.array([-2.0, -6.0])
        Aineq = np.array([[1.0, 1.0], [-1.0, 2.0], [2.0, 1.0]])
        bineq = np.array([2.0, 2.0, 3.0])
        problem = NLPTraced(
            QuadraticProgram(H=H, g=g, Aineq=Aineq, bineq=bineq),
            max_evaluate=54 * FACTOR,
        )

        problem.getInitializationSample = lambda: np.zeros(2)
        x = _solve(problem)
        solution = np.array([0.6667, 1.3333])
        check_sol(problem, x, solution)

    def testHalfcircle(self):
        problem = NLPTraced(HalfCircle(), max_evaluate=64 * FACTOR)
        x = _solve(problem)
        solution = np.array([0, -1.0])
        check_sol(problem, x, solution)

    def testQuadraticIneq3(self):
        """ """
        H = np.array([[100.0, 0.0], [0.0, 1.0]])
        g = np.array([0.0, 0.0])
        Aineq = np.array([[-1.0, 0], [0.0, -1.0]])
        bineq = np.array([-0.1, -0.1])
        problem = NLPTraced(
            QuadraticProgram(H=H, g=g, Aineq=Aineq, bineq=bineq),
            max_evaluate=320 * FACTOR,
        )

        problem.getInitializationSample = lambda: 4 * np.ones(2)
        x = _solve(problem)
        solution = np.array([0.1, 0.1])
        check_sol(problem, x, solution)

    def testLogisticBounds(self):
        problem = NLPTraced(LogisticWithBounds(), 60 * FACTOR)
        x = _solve(problem)
        solution = np.array([2, 2, 1.0369])
        check_sol(problem, x, solution)

    def testQuadraticB(self):
        n = 3
        H = np.array([[1.0, -1.0, 1], [-1, 2, -2], [1, -2, 4]])
        g = np.array([2, -3, 1])
        Aineq = np.vstack((np.identity(n), -np.identity(n)))
        bineq = np.concatenate((np.ones(n), np.zeros(n)))
        Aeq = np.ones(3).reshape(1, -1)
        beq = np.array([0.5])
        problem = NLPTraced(
            QuadraticProgram(H=H, g=g, Aeq=Aeq, beq=beq, Aineq=Aineq, bineq=bineq),
            142 * FACTOR,
        )
        x = solve(problem)
        solution = np.array([0, 0.5, 0])
        check_sol(problem, x, solution)

    def test_nonlinearA(self):
        problem = NLPTraced(NonlinearA(), 307 * FACTOR)
        x = solve(problem)
        solution = np.array([1.00000000, 0])
        check_sol(problem, x, solution)

    def test_f_r(self):
        Q = np.array([[1000.0, 0.0], [0.0, 1.0]])
        R = np.ones((2, 2))
        d = np.zeros(2)
        A = np.array([[1.0, 1.0], [1.0, 0.0]])
        b = np.zeros(2)

        problem = NLPTraced(F_R_Eq(Q, R, d, A, b), 21 * FACTOR)
        x = solve(problem)
        solution = np.array([0.0, 0.0])
        check_sol(problem, x, solution)

    def testQuaterCircle(self):
        problem = NLPTraced(QuarterCircle(), MAX_IT)
        x = solve(problem)
        solution = np.array([0, 0.0])
        check_sol(problem, x, solution)

    def testHalfcircle2(self):
        problem = NLPTraced(HalfCircle2(), MAX_IT)
        x = solve(problem)
        solution = np.array([0.44721368, -0.89442729])
        check_sol(problem, x, solution)

    def testQuadraticC(self):
        n = 3
        H = np.array([[1.0, -1.0, 1], [-1, 2, -2], [1, -2, 4]])
        g = np.array([2, -3, 1])
        Aineq = np.vstack((np.identity(n), -np.identity(n)))
        bineq = np.concatenate((np.ones(n), np.zeros(n)))
        Aeq = np.ones(3).reshape(1, -1)
        beq = np.array([0.5])

        problem = NLPTraced(
            QuadraticProgram(H=H, g=g, Aeq=Aeq, beq=beq, Aineq=Aineq, bineq=bineq),
            MAX_IT,
        )

        problem.getInitializationSample = lambda: np.zeros(3)
        x = solve(problem)
        solution = np.array([0, 0.5, 0])
        check_sol(problem, x, solution)

    def testLogisticBounds_B(self):
        problem = NLPTraced(LogisticWithBounds(), MAX_IT)
        problem.getInitializationSample = lambda: 2 * np.ones(3)
        x = _solve(problem)
        solution = np.array([2, 2, 1.0369])
        check_sol(problem, x, solution)

    def test_nonlinearB(self):
        problem = NLPTraced(NonlinearA(), MAX_IT)
        problem.getInitializationSample = lambda: np.array([1.1, 0.1])
        x = solve(problem)
        solution = np.array([1.00000000, 0])
        check_sol(problem, x, solution)


    def testQuadraticNonConvex(self):
        n = 2
        H = np.array([[-2, 0], [0, 1]])
        g = np.array([0, 0])
        Aineq = np.vstack((np.identity(n), -np.identity(n)))
        bineq = np.concatenate((np.ones(n), np.zeros(n)))

        problem = NLPTraced(
            QuadraticProgram(H=H, g=g,
                             Aineq=Aineq, bineq=bineq),
            MAX_IT,
        )

        problem.getInitializationSample = lambda: .5 * np.ones(2)
        x = solve(problem)
        solution = np.array([1.,0.])
        check_sol(problem, x, solution)



# Run tests with:
# python3 test.py

# Too see help and options
# python3 test.py --help

if __name__ == "__main__":
    unittest.main()
