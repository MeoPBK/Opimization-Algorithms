import unittest
import sys
import numpy as np

sys.path.append("../..")
from solution import DiffOpt, QP_0, QP_1, QP_2


class testProblem(unittest.TestCase):

    problem = DiffOpt

    def set_(self, problem):
        self.problem = problem

    def test_QP0(self):

        QP = QP_0
        c = np.ones(2)
        problem = DiffOpt(c, QP["A"], QP["b"])
        x = QP["x"]
        out = problem.evaluate(x)
        fsol = np.array([1.])
        Jsol = np.array([[1., 1.]])
        self.assertTrue(np.linalg.norm(out[0] - fsol) < 1e-3)
        self.assertTrue(np.linalg.norm(out[1] - Jsol) < 1e-3)

    def test_QP1(self):

        QP = QP_1
        c = np.ones(2)
        problem = DiffOpt(c, QP["A"], QP["b"])
        x = QP["x"]
        out = problem.evaluate(x)
        fsol = np.array([1.])
        Jsol = np.array([[0., 1.]])
        self.assertTrue(np.linalg.norm(out[0] - fsol) < 1e-3)
        self.assertTrue(np.linalg.norm(out[1] - Jsol) < 1e-3)

    def test_QP2(self):

        QP = QP_2
        c = np.ones(2)
        problem = DiffOpt(c, QP["A"], QP["b"])
        x = QP["x"]
        out = problem.evaluate(x)

        fsol = np.array([2.])
        Jsol = np.array([[0., 0.]])
        self.assertTrue(np.linalg.norm(out[0] - fsol) < 1e-3)
        self.assertTrue(np.linalg.norm(out[1] - Jsol) < 1e-3)


# Run tests with:
# python3 test.py
# Too see help and options
# python3 test.py --help
if __name__ == "__main__":
    unittest.main()
