import sys  # noqa
sys.path.append("../..")  # noqa

import unittest
import numpy as np
from solution import solve
from optalg.example_nlps.linear_program_ineq import LinearProgramIneq
from optalg.example_nlps.quadratic_program import QuadraticProgram
from optalg.example_nlps.quarter_circle import QuaterCircle
from optalg.example_nlps.halfcircle import HalfCircle
from optalg.interface.nlp_traced import NLPTraced
from optalg.interface.objective_type import OT


MAX_EVALUATE = 10000


# def _solve(nlp):
#     return run_with_timeout(lambda: solve(nlp))

def _solve(nlp):
    return solve(nlp)


class testLogBarrier(unittest.TestCase):

    def testLinear1(self):

        problem = NLPTraced(LinearProgramIneq(2), max_evaluate=MAX_EVALUATE)
        x = _solve(problem)
        solution = np.zeros(2)
        fout = problem.evaluate(x)[0][0]
        fsol = problem.evaluate(solution)[0][0]
        id_ineq = [i for i, t in enumerate(
            problem.getFeatureTypes()) if t == OT.ineq]
        self.assertTrue(np.sum(problem.evaluate(x)[0][id_ineq] > 0) == 0)
        self.assertTrue(fout < fsol + 1e-3)

    def testLinear2(self):

        problem = NLPTraced(LinearProgramIneq(20), max_evaluate=MAX_EVALUATE)
        problem.getInitializationSample = lambda: .1 + .1 * np.arange(20)
        x = _solve(problem)
        solution = np.zeros(20)
        fout = problem.evaluate(x)[0][0]
        fsol = problem.evaluate(solution)[0][0]
        id_ineq = [i for i, t in enumerate(
            problem.getFeatureTypes()) if t == OT.ineq]
        self.assertTrue(np.sum(problem.evaluate(x)[0][id_ineq] > 0) == 0)
        self.assertTrue(fout < fsol + 1e-3)

    def testQuadraticIneq(self):
        """
        """
        H = np.array([[1., -1.], [-1., 2.]])
        g = np.array([-2., -6.])
        Aineq = np.array([[1., 1.], [-1., 2.], [2., 1.]])
        bineq = np.array([2., 2., 3.])
        problem = NLPTraced(
            QuadraticProgram(
                H=H,
                g=g,
                Aineq=Aineq,
                bineq=bineq),
            max_evaluate=MAX_EVALUATE)

        problem.getInitializationSample = lambda: np.zeros(2)
        x = _solve(problem)
        solution = np.array([0.6667, 1.3333])
        fout = problem.evaluate(x)[0][0]
        fsol = problem.evaluate(solution)[0][0]
        id_ineq = [i for i, t in enumerate(
            problem.getFeatureTypes()) if t == OT.ineq]
        self.assertTrue(np.sum(problem.evaluate(x)[0][id_ineq] > 0) == 0)
        self.assertTrue(fout < fsol + 1e-3)

    def testHalfcircle(self):
        problem = NLPTraced(HalfCircle(), max_evaluate=MAX_EVALUATE)
        x = _solve(problem)
        solution = np.array([0, -1.])
        fout = problem.evaluate(x)[0][0]
        fsol = problem.evaluate(solution)[0][0]
        self.assertTrue(fout < fsol + 1e-3)

    def testQuaterCircle(self):
        problem = NLPTraced(QuaterCircle(), max_evaluate=MAX_EVALUATE)
        x = _solve(problem)
        solution = np.array([0, 0.])
        fout = problem.evaluate(x)[0][0]
        fsol = problem.evaluate(solution)[0][0]
        id_ineq = [i for i, t in enumerate(
            problem.getFeatureTypes()) if t == OT.ineq]
        self.assertTrue(np.sum(problem.evaluate(x)[0][id_ineq] > 0) == 0)
        self.assertTrue(fout < fsol + 1e-3)

    def testQuadraticIneq2(self):
        """
        """
        H = np.array([[10., 0.], [0., 1.]])
        g = np.array([1., 1.])
        Aineq = np.array([[1., 1.], [-1., 2.], [2., 1.]])
        bineq = np.array([2., 2., 3.])
        problem = NLPTraced(
            QuadraticProgram(
                H=H,
                g=g,
                Aineq=Aineq,
                bineq=bineq),
            max_evaluate=MAX_EVALUATE)

        problem.getInitializationSample = lambda: np.zeros(2)
        x = _solve(problem)
        solution = np.array([-0.1, -1])
        fout = problem.evaluate(x)[0][0]
        fsol = problem.evaluate(solution)[0][0]
        id_ineq = [i for i, t in enumerate(
            problem.getFeatureTypes()) if t == OT.ineq]
        self.assertTrue(np.sum(problem.evaluate(x)[0][id_ineq] > 0) == 0)
        self.assertTrue(fout < fsol + 1e-3)

    def testQuadraticIneq3(self):
        """
        """
        H = np.array([[100000., 0.], [0., 1.]])
        g = np.array([0., 0.])
        Aineq = np.array([[-1., 0], [0., -1.]])
        bineq = np.array([-0.1, -0.1])
        problem = NLPTraced(
            QuadraticProgram(
                H=H,
                g=g,
                Aineq=Aineq,
                bineq=bineq),
            max_evaluate=MAX_EVALUATE)

        problem.getInitializationSample = lambda: 4 * np.ones(2)
        x = _solve(problem)
        solution = np.array([0.1, 0.1])
        fout = problem.evaluate(x)[0][0]
        fsol = problem.evaluate(solution)[0][0]
        id_ineq = [i for i, t in enumerate(
            problem.getFeatureTypes()) if t == OT.ineq]
        self.assertTrue(np.sum(problem.evaluate(x)[0][id_ineq] > 0) == 0)
        self.assertTrue(fout < fsol + 1e-3)

    def testHalfcircle2(self):
        problem = NLPTraced(HalfCircle(), max_evaluate=MAX_EVALUATE)
        problem.getInitializationSample = lambda: np.array([0.1, .9])
        x = _solve(problem)
        solution = np.array([0, -1.])
        self.assertTrue(np.linalg.norm(solution - x) < 1e-3)

    def testQuadraticIneq4(self):
        """
        """
        H = np.array([[1000., 0.], [0., 1.]])
        g = np.array([0., 0.])
        Aineq = 100. * np.array([[-1., 0], [0., -1.]])
        bineq = 100. * np.array([-0.1, -0.1])
        problem = NLPTraced(
            QuadraticProgram(
                H=H,
                g=g,
                Aineq=Aineq,
                bineq=bineq),
            max_evaluate=MAX_EVALUATE)

        problem.getInitializationSample = lambda: np.array([.3, 1.5])
        x = _solve(problem)
        solution = np.array([0.1, 0.1])
        fout = problem.evaluate(x)[0][0]
        fsol = problem.evaluate(solution)[0][0]
        id_ineq = [i for i, t in enumerate(
            problem.getFeatureTypes()) if t == OT.ineq]
        self.assertTrue(np.sum(problem.evaluate(x)[0][id_ineq] > 0) == 0)
        self.assertTrue(fout < fsol + 1e-3)

    def testLinearIneq(self):
        """
        """
        H = np.zeros((2, 2))
        g = np.array([-30., -10.])
        Aineq = np.array([[6., 3.], [3., -1.], [1., .25], [-1, 0], [0, -1]])
        bineq = np.array([40., 0., 4., 0, 0])
        problem = NLPTraced(
            QuadraticProgram(
                H=H,
                g=g,
                Aineq=Aineq,
                bineq=bineq),
            max_evaluate=MAX_EVALUATE)

        problem.getInitializationSample = lambda: np.array([1., 8.])
        x = _solve(problem)
        print(x)
        solution = np.array([1.3333, 10.6667])
        fout = problem.evaluate(x)[0][0]
        fsol = problem.evaluate(solution)[0][0]
        id_ineq = [i for i, t in enumerate(
            problem.getFeatureTypes()) if t == OT.ineq]
        self.assertTrue(np.sum(problem.evaluate(x)[0][id_ineq] > 0) == 0)
        self.assertTrue(fout < fsol + 1e-3)


if __name__ == "__main__":
    unittest.main()
