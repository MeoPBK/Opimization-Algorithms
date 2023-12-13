import sys  # noqa
sys.path.append("../..")  # noqa

import numpy as np
from solution import NLP_Gaussina_ineq
from optalg.interface.nlp import NLP
from optalg.utils.finite_diff import finite_diff_J, finite_diff_hess


# You can freely modify this script to play around with
# the implementation of your NLP


# Example:
D = np.ones((2, 2))
A = np.ones((2, 2))
b = np.ones(2)
x0 = np.zeros(2)

problem = NLP_Gaussina_ineq(x0, D, A, b)
x = np.ones(2)
y, J = problem.evaluate(x)
H = problem.getFHessian(x)
print(y)
print(J)
print(H)
