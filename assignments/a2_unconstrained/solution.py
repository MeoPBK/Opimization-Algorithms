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

    #
    # Write your code here
    #


    # return found solution
    return x
