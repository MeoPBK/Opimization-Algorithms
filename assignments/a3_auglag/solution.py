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
    idx = [[1 for i, s in enumerate(types) if s == OT.f], [1 for i, s in enumerate(types) if s == OT.r], [1 for i, s in enumerate(types) if s == OT.ineq],[1 for i, s in enumerate(types) if s == OT.eq]]
    lb, k = np.zeros(len(idx[2])), np.zeros(len(idx[3]))
    tol = 0.0005
    mu, nu = 10, 10
    dx = 1

    phi, J = nlp.evaluate(x)
    i = 1
    while(dx >= tol or (np.all(phi[idx[2]]>= tol) and len(idx[2])!=0) or (np.all(np.abs(phi[idx[3]]) >= tol) and len(idx[3])!=0)):
        x_tmp = x.copy()
        x, i = solve_unc(x, lb, k, mu, nu, tol, nlp, idx, i)
        phi, J = nlp.evaluate(x)
        i = i+1
        lb += mu*2*phi[idx[2]]
        lb = np.maximum(lb,0)
        k += nu*2*phi[idx[3]]
        dx = np.linalg.norm(x_tmp-x)

    #


    return x

def my_func(x, k, lb, mu, nu, nlp, idx, hess = False):

    phi, J = nlp.evaluate(x)
    f =phi[idx[0]]
    r = phi[id[1]]
    g =phi[idx[2]]
    h = phi[idx[3]]
    Jf = J[idx[0]]
    Jr = J[idx[1]]
    Jg = J[idx[2]]
    Jh = J[idx[3]]

    cn = np.greater_equal(g,0)
    cn2 = np.greater(lb,0)
    act_cn = np.logicsl_or(cn, cn2)
    phi = (np.sum(r**2)+np.inner(g,lb)+mu*np.inner(g**2,act_cn)+np.inner(h,k)+nu*np.sum(h**2)+np.sum(r**2))
    J = (Jf + 2*nu*h@Jh + 2*mu*(g*act_cn)@Jg + (r@Jr*2)+(k@Jh)+(lb@Jg))
    J = J.copy()
    if hess is False:
        return phi, J
    else:
        H = nlp.getFHessian(x).copy()
        if (H.dtype==np.int64):
            H = H.astype(np.float64)
        if len(idx[1]) !=0:
            H += 2*(Jr.T@Jr)
        if len(idx[2]) !=0:
            Jg = Jg*np.reshape(act_cn, newshape: (len(idx[2]),1))
            tmp = mu*2*(Jg.T@Jg)
            H+=tmp
        if len(idx[3]) != 0:
            H += nu*2*(Jh.T@Jh)
        return phi, J, H

def solve_unc(x, lb, k, mu, nu, tol, nlp, idx, i):
    i += 1
    a_plus = 1.2
    a_minus = 0.5
    ls = 0.5
    delta = 1
    a = 1

    phi, J, H = my_func(x, k, lb, mu, nu, nlp, idx, hess: True)

    while (np.linalg.norm(delta*a)>tol):
        i +=1
        delta = -J[0]
        if not np.isclose(np.linalg.det(H), b:0):
            tmp = np.linalg.solve(H,delta)
            if J[0].dot(tmp) <= 0:
                delta = tmp
        i += 1
        phi_tmp, J_tmp, H = my_func(x+a*delta, k, lb, mu, nu, nlp, idx, hess: True)
        while phi_tmp > phi+ls*J[0].T.dot(a*delta):
            a =a_minus*a
            phi_tmp, J_tmp, H = my_func(x + a * delta, k, lb, mu, nu, nlp, idx, hess: True)
            i = i+1
        x += a*delta
        a = min(a_plus*a,1)
        phi, J = phi_tmp, J_tmp
    return x, i