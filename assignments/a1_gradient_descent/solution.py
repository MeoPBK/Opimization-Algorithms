import sys
sys.path.append("../..")
from optalg.interface.objective_type import OT
from optalg.interface.nlp import NLP
import numpy as np

# cancel later:
from optalg.example_nlps.quadratic_identity_2 import QuadraticIdentity2
from optalg.interface.nlp_traced import NLPTraced
def solve(nlp: NLP):
    """
    Gradient descent with backtracking Line search


    Arguments:
    ---
        nlp: object of class NLP that only contains one feature of type OT.f.

    Returns:
        x: local optimal solution (1-D np.ndarray)


    Task:
    ---

    Implement a solver that does iterations of gradient descent
    with a backtracking line search

    x = x - k * Df(x),

    where Df(x) is the gradient of f(x)
    and the step size k is computed adaptively with backtracking line search

    Notes:
    ---

    Get the starting point with:
    x = nlp.getInitializationSample()

    Use the following to query the problem:
    phi, J = nlp.evaluate(x)

    phi is a vector (1D np.ndarray); use phi[0] to access the cost value
    (a float number).

    J is a Jacobian matrix (2D np.ndarray). Use J[0] to access the gradient
    (1D np.array) of phi[0].

    """

    # sanity check on the input nlp
    assert len(nlp.getFeatureTypes()) == 1
    assert nlp.getFeatureTypes()[0] == OT.f

    # get start point
    x = np.copy(nlp.getInitializationSample())

    # Add your code here

    ## Parameters
    k = 1
    rho_ls = 0.01
    rho_a_plus = 1.2
    rho_a_min = 0.5

    ## Parameters (not sure)
    tol=0.001
    delta_max = np.inf
    ## INITIALIZATION
    phi, J = nlp.evaluate(x)
    ## DEBUG check values
   # i = 0
   # print("i:", i)
   # print("alpha:", k)
   # print("x:", x)
   # print("phi(x):", phi)
   # print("J:", J)

    while(np.linalg.norm(-k*J[0])>=tol):
        ## gradient descent
        x_plus = -k * J[0] + x
        phi_new, J_new = nlp.evaluate(x_plus)
        ## new conditions
        cond = phi[0] + rho_ls * np.dot(J[0], (-k) * J[0])
        ## backtracking line search
        while phi_new[0]>(cond):
            k = rho_a_min * k           # minimize k (alpha)
            x_plus = -k * J[0] + x
            phi_new, J_new = nlp.evaluate(x_plus)
            cond = phi[0] + rho_ls * np.dot(J[0], (-k) * J[0])
        ## reasign variable values
        phi = phi_new
        J = J_new
        x = x_plus
        ## riasign k (alpha) value
        k = min(rho_a_plus*k,delta_max)

        ## DEBUG check values
     #   i = i+1
     #   print("i:",i)
     #   print("alpha:",k)
     #   print("x:",x)
        print("phi(x):",type(phi))
     #   print("J:",J)
    return x


