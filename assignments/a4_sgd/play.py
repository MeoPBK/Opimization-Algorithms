import sys
sys.path.append("../..")
from optalg.interface.objective_type import OT
from optalg.interface.nlp import NLP
import numpy as np
from optalg.example_nlps.quadratic_stochastic import Quadratic_stochastic
from optalg.example_nlps.linear_least_squares_stochastic import Linear_least_squares_stochastic
from solution import solve


A = np.array([[0.85880983, 0.42075752, 0.14625862],
              [0.57705246, 0.76635021, 0.64077446],
              [0.85916002, 0.86186594, 0.81231712],
              [1., 0., 0.],
              [0., 1., 0.],
              [0., 0., 1.]])

b = np.zeros(6)

problem = Linear_least_squares_stochastic(A, b)

x = solve(problem)
print("output", x)
print("otpimal solution", problem.opt)
