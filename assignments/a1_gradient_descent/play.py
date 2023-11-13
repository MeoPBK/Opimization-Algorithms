import sys
sys.path.append("../..")

from solution import solve
from optalg.example_nlps.quadratic_identity_2 import QuadraticIdentity2
from optalg.interface.nlp_traced import NLPTraced
import numpy as np
import unittest
from optalg.utils.timeout import run_with_timeout
from optalg.example_nlps.barrier import Barrier
from optalg.example_nlps.hole import Hole

# import matplotlib as plt

# You can freely modify this script to play around with your solver
# and the problems we provide in test.py
# Run the script with
# python3 play.py


# Example:
print("QAUDRATIC FCT")
problem = NLPTraced(QuadraticIdentity2())
b = vars(problem)
a = dir(problem)
x = solve(problem)

print("================================")
print("BARRIER")
problem = NLPTraced(Barrier())
solution = 0.01 * np.ones(2)
#x = run_with_timeout(lambda: solve(problem))
x = solve(problem)
print("found solution", x)
print("real solution", solution)
fout = problem.evaluate(x)[0][0]
fopt = problem.evaluate(solution)[0][0]
if(np.linalg.norm(fout - fopt) < 1e-3):
    print("precision passed")
else:
    print("precision failed")

print("fout",fout,"-fopt",fopt,"=")
print(fout-fopt)

print("===================================")
print("QAUDRATIC FCT")

problem = NLPTraced(QuadraticIdentity2())

x = solve(problem)
fout = problem.evaluate(x)[0][0]
solution = np.zeros(2)
fopt = problem.evaluate(solution)[0][0]

print("found solution", x)
print("real solution",solution)

if(np.linalg.norm(fout - fopt) < 1e-3):
    print("precision passed")
else:
    print("precision failed")
print("fout",fout,"-fopt",fopt,"=")
print(fout-fopt)
print("==================================")
print("HOLE")


def make_C_exercise1(n, c):
    """
    n: integer
    c: float
    """
    C = np.zeros((n, n))
    for i in range(n):
        C[i, i] = c ** (float(i - 1) / (n - 1))
    return C


C = make_C_exercise1(3, .1)
problem = NLPTraced(Hole(C, 1.5))
solution = np.zeros(3)
x = solve(problem)
#x = run_with_timeout(lambda: solve(problem))
fout = problem.evaluate(x)[0][0]
fopt = problem.evaluate(solution)[0][0]
print("found solution", x)
print("real solution",solution)

if(np.linalg.norm(fout - fopt) < 1e-3):
    print("precision passed")
else:
    print("precision failed")
print("fout",fout,"-fopt",fopt,"=")
print(fout-fopt)
#test.py --help