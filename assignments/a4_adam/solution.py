import sys
sys.path.append("../..")
from optalg.interface.nlp_stochastic import NLP_stochastic
import numpy as np


def solve(nlp: NLP_stochastic):
    """
    Stochastic gradient descent -- ADAM


    Arguments:
    ---
        nlp: object of class NLP_stochastic that contains one feature of type OT.f.

    Returns:
    ---
        x: local optimal solution (1-D np.ndarray)

    Task:
    ---
    See the coding assignment PDF in ISIS.

    Notes:
    ---

    Get the starting point with:
    x = nlp.getInitializationSample()

    Get the number of samples with:
    N = nlp.getNumSamples()

    You can query the problem with any index i=0,...,N (N not included)

    y, J = nlp.evaluate_i(x, i)

    As usual, get the cost function (scalar) and gradient (1-D np.array) with y[0] and J[0]

    The output (y,J) is different for different values of i and x.

    The expected value (over i) of y,J at a given x is SUM_i [ nlp.evaluate_i(x, i) ]  / N

    """

    x = nlp.getInitializationSample()
    N = nlp.getNumSamples()

    #
    a = 0.1
    b1 = 0.9
    b2 = 0.9
    s = 0
    m =np.zeros_like(x)
    z = m
    tol = 1e-8
    theta = tol
    count = 0
    chk = False

    i = 0
    j = 1

    idx =np.arange(0,N)

    while(True):
        for l in np.random.permutation(idx):
            s = s+1
            count += 1
            phi, J = nlp.evaluate_i(x,l)

            m = m*b1+(1-b1)*J[0]
            z = z*b2+(1-b2)*np.square(J[0])

            m = m/(1-b1**s)
            z = z/(1-b2**s)

            x_tmp = x-a*m/(np.sqrt(z)+tol)

            if count > 9999 or np.dot(J[0],J[0])<= tol:
                i +=1
                if i>=j or count > 9999:
                    chk = True
                    break
                else:
                    i =0
        if chk:
            break
    #


    return x
