import sys
sys.path.append("../..")

from solution import NLP_xCCx
from optalg.interface.nlp import NLP
from optalg.utils.finite_diff import finite_diff_J, finite_diff_hess
import numpy as np
import unittest
from optalg.interface.nlp_timeout import NLPTimeout



# You can freely modify this script to play around with
# the implementation of your NLP


# Example:
C = np.ones((2, 2))
problem = NLP_xCCx(C)
x = np.ones(2)
value = problem.evaluate(x)[0][0]
solution = 8

C = np.ones((2, 2))
problem = NLP_xCCx(C)
x = np.array([-1, .5])
_, J = problem.evaluate(x)
eps = 1e-5
Jdiff = finite_diff_J(problem, x, eps)
print(J)
print(Jdiff)

print("found solution", value)
print("real solution", solution)
