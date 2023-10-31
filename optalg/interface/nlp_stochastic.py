import numpy as np
from .nlp import NLP


class NLP_stochastic(NLP):

    def evaluate(self, x: np.ndarray):
        raise NotImplementedError()

    def evaluate_i(self, x: np.ndarray, i: int):
        raise NotImplementedError()

    def getNumSamples(self):
        return -1
