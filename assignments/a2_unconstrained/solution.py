import sys  # noqa
sys.path.append("../..")  # noqa

import numpy as np
from optalg.interface.nlp import NLP
from optalg.interface.objective_type import OT


def solve(nlp: NLP, Dout={}):
    """
    Solver for unconstrained optimization


    Arguments:
    ---
        nlp: object of class NLP that only contains one feature of type OT.f.

    Returns:
        x: local optimal solution (1-D np.ndarray)


    Task:
    ---

    See instructions and requirements in the coding assignment PDF in ISIS.

    Notes:
    ---

    Get the starting point with:
    x = nlp.getInitializationSample()

    Use the following to query the function and gradient of the problem:
    phi, J = nlp.evaluate(x)

    phi is a vector (1D np.ndarray); use phi[0] to access the cost value
    (a float number).

    J is a Jacobian matrix (2D np.ndarray). Use J[0] to access the gradient
    (1D np.array) of phi[0].

    Use getFHessian to query the Hessian.

    H = nlp.getFHessian(x)

    H is a matrix (2D np.ndarray) of shape n x n.


    You can use Dout to store any information you want during the computation,
    to analyze and debug your code.
    For instance, you can store the value of the cost function at each
    iteration, the variable x,...


    Dout["xs"] = []
    Dout["f"] = []

    ...

    Dout["x"].append(np.copy(x))
    Dout["f"].append(f)


    Do not use the assignment operator Dout = { ... }  in your code,
    just use Dout[...] = ...,
    otherwise you will not be able to access the information
    outside the solver.

    In test file, we call solve only with one argument, but it is fine
    if your code actually stores information in Dout.

    """

    # sanity check on the input nlp
    assert len(nlp.getFeatureTypes()) == 1
    assert nlp.getFeatureTypes()[0] == OT.f

    # get start point
    x = nlp.getInitializationSample()

    # Comment/Uncomment if you want to store information in Dout
    Dout["xs"] = []
    Dout["x0"] = np.copy(x)
    Dout["xs"].append(np.copy(x))

    print("x_u: ", x)
    ## Gradient & Hessian
    phi, J = nlp.evaluate(x)    # phi[0],J[0]
    H = nlp.getFHessian(x)
    Dout["f"] = []
    Dout["f"].append(np.copy(phi))

    ## Parameters
    k = 1           # alpha
    lb = 1          # lambda
    rho_ls = 0.01
    rho_a_plus = 1.2
    rho_a_min = 0.5
    rho_lb_plus = 1.2
    rho_lb_min = 0.5
    tol = 0.0001
    delta_max = np.inf

    delta = np.linalg.solve((H + lb * np.identity(len(H))), -J[0])

    i = 0
    while (np.linalg.norm(k*delta,np.inf) >= tol):
        ## Newton Method
        if J[0].T@delta > 0:
            #delta = -J[0]
            # if -np.min(np.linalg.eig(H)) > 0:
            lb = -np.min(np.linalg.eigvals(H)) + 0.001
            delta = np.linalg.solve((H+lb*np.identity(len(H))),-J[0])

        x_plus = k*delta + x
        phi_new, J_new = nlp.evaluate(x_plus)

        ## new conditions
        cond = phi[0] + rho_ls * np.dot(J[0], k*delta)

        ## backtracking line search
        while phi_new[0] > (cond):
            k = rho_a_min * k  # minimize k (alpha)
            lb = rho_lb_plus*lb  # maximize lambda

            delta = np.linalg.solve((H + lb * np.identity(len(H))), -J[0])  # recompute delta
            i +=1
            x_plus = x + k*delta
            phi_new, J_new = nlp.evaluate(x_plus)
            cond = phi[0] + rho_ls * np.dot(J[0], k*delta)

        ## reasign variable values
        phi = phi_new
        J = J_new
        x = x_plus
        H = nlp.getFHessian(x_plus)
        i += 1
        Dout["xs"].append(np.copy(x))
        Dout["f"].append(phi)

        ## riasign k (alpha) and lb (lambda) value
        k = min(rho_a_plus * k, 1)
        lb = rho_lb_min*lb
        delta = np.linalg.solve((H+lb*np.identity(len(H))),-J[0])

    # return found solution
    print("i: ",i)
    return x
