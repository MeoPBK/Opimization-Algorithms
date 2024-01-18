import sys

sys.path.append("../..")

import numpy as np
from optalg.example_nlps.linear_program_ineq import LinearProgramIneq
from optalg.example_nlps.quadratic_program import QuadraticProgram
from optalg.example_nlps.quarter_circle import QuarterCircle
from optalg.example_nlps.halfcircle import HalfCircle
from optalg.example_nlps.logistic_bounds import LogisticWithBounds
from optalg.example_nlps.nonlinearA import NonlinearA
from optalg.interface.nlp_traced import NLPTraced
from optalg.example_nlps.f_r_eq import F_R_Eq
from solution import *
from optalg.utils.finite_diff import *

problem = LinearProgramIneq(2)
Dout = {}
x = solve(problem, Dout)
solution = np.zeros(2)

print("x", x)
print("solution", solution)
