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
    ## create empty containers
    Dout["xs"] = []
    Dout["x0"] = np.copy(x)
    Dout["xs"].append(np.copy(x))
    Dout["f"] = []
    Dout["g"] = []
    Dout["B"] = []
    Dout["mu"] = []

    ## PARAMETERS
    mu = 1
    rho_mu_min = 0.5
    rho_ls = 0.01
    rho_a_plus = 1.2
    rho_a_min = 0.5
    rho_lb_plus = 1.2
    rho_lb_min = 0.5
    tol = 0.0001
    i = 0
    x_p = [np.inf]*len(x)
    types = nlp.getFeatureTypes()
    id_f = [i for i, t in enumerate(types) if t == OT.f]
    id_ineq = [i for i, t in enumerate(types) if t == OT.ineq]
    iinf= 10000000
    chk = True

    ## first Hessian evaluation
    H = nlp.getFHessian(x)
    y, J = nlp.evaluate(x)

    while(np.linalg.norm(x-x_p)>= tol):
        ## Parameters
        k = 1  # alpha
        lb = 1  # lambda
        x_p = x

        ## evaluate function
        y, J = nlp.evaluate(x)
        B = y[id_f[0]] - mu*np.sum(np.log(-y[1:]))
        #tmp = [np.outer(J[k].T, J[k]) for k in id_ineq]
        #Hesse = H + mu * np.sum([tmp[j-1] / (y[j] ** 2) for j in id_ineq], axis=0)
        Hesse= H + mu*np.sum(1/y[1:,None,None]**2*np.dot(J[1:].T,J[1:]),axis = 0)
        Jakob = J[id_f[0]]-mu*np.sum([J[j]/y[j] for j in id_ineq],axis=0)
        delta = np.linalg.solve((Hesse + lb * np.identity(len(Hesse))), -Jakob)

        ## pass values to containers
        Dout["f"].append(np.copy(y[id_f[0]]))
        Dout["g"].append(np.copy(y[id_ineq]))
        Dout["B"].append(np.copy(B))
        Dout["mu"].append(np.copy(mu))

        print("mu: ", mu)
        i += 1

        ## Newton Method
        while (np.linalg.norm(k * delta, np.inf) >= tol):
            if Jakob.T @ delta > 0:     ## check for pos. definiteness
                # delta = -J[0]
                lb = -np.min(np.linalg.eigvals(Hesse)) + 0.001
                delta = np.linalg.solve((Hesse + lb * np.identity(len(Hesse))), -Jakob)

            ## evaluate function for Wolf conditions
            x_plus = k * delta + x
            y_new, J_new = nlp.evaluate(x_plus)
            if y_new[id_ineq].all() < 0 or chk is True:
                B_new = y_new[id_f[0]] - mu * np.sum(np.log(-y_new[1:]))
            else:
                B_new = iinf
            ## new conditions
            cond = B + rho_ls * np.dot(Jakob, k * delta)

            ## backtracking line search
            while B_new > (cond):
                ## compute new parameters
                k = rho_a_min * k  # minimize k (alpha)
                lb = rho_lb_plus * lb  # maximize lambda

                # compute new Wolf conditions and evaluate function
                delta = np.linalg.solve((Hesse + lb * np.identity(len(Hesse))), -Jakob)  # recompute delta
                x_plus = x + k * delta
                y_new, J_new = nlp.evaluate(x_plus)
                cond = B + rho_ls * np.dot(Jakob, k * delta)

                if y_new[id_ineq].all() < 0 or chk is True:
                    B_new = y_new[id_f[0]] - mu * np.sum(np.log(-y_new[1:]))
                else:
                    B_new = iinf
                    #print("ff")

                i += 1

            ## reassign variable values
            B = B_new
            J = J_new
            x = x_plus
            ## evaluate function
            Jakob = J[id_f[0]] - mu * np.sum([J[j] / y[j] for j in id_ineq], axis=0)
            H = nlp.getFHessian(x)
            #tmp = [np.outer(J[k].T,J[k]) for k in id_ineq]
            #Hesse = H + mu * np.sum([tmp[j-1] / (y[j] ** 2) for j in id_ineq], axis=0)
            Hesse = H + mu * np.sum(1 / y[1:, None, None] ** 2 * np.dot(J[1:].T, J[1:]), axis=0)
            i += 1

            ## pass values to containers
            Dout["xs"].append(np.copy(x))
            Dout["f"].append(np.copy(y[id_f[0]]))
            Dout["g"].append(np.copy(y[id_ineq]))
            Dout["B"].append(np.copy(B))
            Dout["mu"].append(np.copy(mu))

            ## compute k (alpha) and lb (lambda) value
            k = min(rho_a_plus * k, 1)
            lb = rho_lb_min * lb

        ## compute new parameter mu
        mu = rho_mu_min*mu
        i += 1
    print("i_log: ",i)
    #print("g(x_out): ", y[id_ineq])
    #

    return x
