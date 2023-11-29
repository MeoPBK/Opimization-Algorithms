import sys  # noqa
sys.path.append("../..")  # noqa

import numpy as np
from optalg.interface.objective_type import OT
from optalg.interface.nlp import NLP






def solve(nlp: NLP, Dout={}):
    """
    solver for constrained optimization (cost term and inequalities)


    Arguments:
    ---
        nlp: object of class NLP that contains one feature of type OT.f,
            and m features of type OT.ineq.

    Returns:
        x: local optimal solution (1-D np.ndarray)


    Task:
    ---
    See the coding assignment PDF in ISIS.


    Notes:
    ---

    Get the starting point with:

    x = nlp.getInitializationSample()

    To know which type (cost term or inequalities) are the entries in
    the feature vector, use:
    types = nlp.getFeatureTypes()

    Index of cost term
    id_f = [ i for i,t in enumerate(types) if t == OT.f ]
    There is only one term of type OT.f ( len(id_f) == 1 )

    Index of inequality constraints:
    id_ineq = [ i for i,t in enumerate(types) if t == OT.ineq ]

    Get all features (cost and constraints) with:

    y,J = nlp.evaluate(x)
    H = npl.getFHessian(x)

    The value, gradient and Hessian of the cost are:

    y[id_f[0]] (scalar), J[id_f[0]], H

    The value and Jacobian of inequalities are:
    y[id_ineq] (1-D np.array), J[id_ineq]


    You can use Dout to store any information you want during the computation,
    to analyze and debug your code -- See the comments in
    a2_unconstrained/solution.py for details.

    """

    x = nlp.getInitializationSample()

    #
    # Write your code Here
    #


    return x
