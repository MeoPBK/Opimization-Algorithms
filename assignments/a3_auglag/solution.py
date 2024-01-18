import sys
sys.path.append("../..")
from optalg.interface.objective_type import OT
from optalg.interface.nlp import NLP
import numpy as np






def solve(nlp: NLP, Dout={}):
    """
    solver for constrained optimization


    Arguments:
    ---
        nlp: object of class NLP that contains features of type OT.f, OT.r, OT.eq, OT.ineq

    Returns:
        x: local optimal solution (1-D np.ndarray)


    Task:
    ---
    See the coding assignment PDF in ISIS.


    Notes:
    ---

    Get the starting point with:

    x = nlp.getInitializationSample()


    You can query the problem with:

    y,J = nlp.evaluate(x)
    H = npl.getFHessian(x)

    To know which type (normal cost, least squares, equalities or inequalities) are the entries in
    the feature vector, use:

    types = nlp.getFeatureTypes()

    id_f = [i for i, t in enumerate(types) if t == OT.f]
    id_r = [i for i, t in enumerate(types) if t == OT.r]
    id_ineq = [i for i, t in enumerate(types) if t == OT.ineq]
    id_eq = [i for i, t in enumerate(types) if t == OT.eq]

    Note that getFHessian(x) only returns the Hessian of the term of type OT.f.

    For example, you can get the value and Jacobian of the equality constraints with y[id_eq] (1-D np.array), and J[id_eq] (2-D np.array).

    All input NLPs contain one feature of type OT.f (len(id_f) is 1). In some problems,
    there no equality constraints, inequality constraints or residual terms.
    In those cases, some of the lists of indexes will be empty (e.g. id_eq = [] if there are not equality constraints).

    You can use Dout to store any information you want during the computation,
    to analyze and debug your code -- See the comments in
    a2_unconstrained/solution.py for details.

    """

    x = nlp.getInitializationSample()

    types = nlp.getFeatureTypes()

    #
    # Write your code Here
    #


    return x
