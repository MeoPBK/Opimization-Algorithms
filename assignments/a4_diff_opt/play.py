import sys
import numpy as np

sys.path.append("../..")
from optalg.utils.finite_diff import *
from solution import DiffOpt, QP_0


QP = QP_0
c = np.ones(2)
problem = DiffOpt(c, QP["A"], QP["b"])
x = .5 * np.ones(2)
out = problem.evaluate(x)
print("computed", out)
print("expected", np.array([1.]), np.array([[1., 1.]]))
