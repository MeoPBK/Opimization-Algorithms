import sys
sys.path.append("../..")

from optalg.example_nlps.logistic import Logistic
from optalg.interface.nlp import NLP
from optalg.utils.finite_diff import *
import numpy as np
import unittest
import math


class testLogistic(unittest.TestCase):
    """
    test mathematical program Logistic
    """
    problem = Logistic

    def testConstructor(self):
        p = self.problem()

    def testValue1(self):
        problem = self.problem()
        phi, J = problem.evaluate(problem.xopt)
        self.assertTrue(np.allclose(phi, np.zeros(problem.num_points)))

    def testJacobian(self):
        problem = self.problem()
        x = np.array([-1, .5, 1])
        flag, _, _ = check_nlp(
            problem.evaluate, x, 1e-5, True)
        self.assertTrue(flag)


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
