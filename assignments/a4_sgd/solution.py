import sys
sys.path.append("../..")
from optalg.interface.nlp_stochastic import NLP_stochastic
import numpy as np


def solve(nlp: NLP_stochastic):
    """
    stochastic gradient descent


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
    # Write your code Here
    #

    return x
