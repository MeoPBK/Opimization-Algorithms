import sys
sys.path.append("../..")

from solution import NLP_xCCx
from optalg.interface.nlp import NLP
from optalg.utils.finite_diff import finite_diff_J, finite_diff_hess
import numpy as np
import unittest


# You can freely modify this script to play around with
# the implementation of your NLP


# Example:
C = np.ones((2, 2))
problem = NLP_xCCx(C)
x = np.ones(2)
y, J = problem.evaluate(x)
value = y[0]
solution = 8


print("found solution", value)
print("real solution", solution)
