import sys  # noqa
sys.path.append("../..")  # noqa

import numpy as np
from solution import solve
from optalg.example_nlps.hole import Hole
from optalg.example_nlps.barrier import Barrier
from optalg.example_nlps.rosenbrock import Rosenbrock
from optalg.example_nlps.rosenbrockN import RosenbrockN
from optalg.example_nlps.cos_lsqs import Cos_lsqs
from optalg.example_nlps.quadratic import Quadratic
from optalg.interface.nlp_traced import NLPTraced


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
Dout = {}
x = solve(problem, Dout)
print(Dout)

print(x)
print(solution)
