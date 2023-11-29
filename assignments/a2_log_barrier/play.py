import sys  # noqa
sys.path.append("../..")  # noqa

import numpy as np
from optalg.interface.nlp import NLP
from optalg.interface.objective_type import OT
from optalg.example_nlps.linear_program_ineq import LinearProgramIneq
from solution import solve

problem = LinearProgramIneq(2)
solution = np.zeros(2)
Dout = {}
x = solve(problem, Dout)
print(Dout)
print(x)
print(solution)
