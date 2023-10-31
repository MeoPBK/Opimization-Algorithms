from .nlp import NLP
from ..utils.timeout import run_with_timeout


class NLPTimeout(NLP):

    def __init__(self, nlp, timeout=2):
        self.nlp = nlp
        self.timeout = timeout

        super().__init__()

    def evaluate(self, x):
        """
        """
        return run_with_timeout(
            lambda: self.nlp.evaluate(x),
            seconds=self.timeout)

    def getBounds(self):
        """
        """
        return self.nlp.getBounds()

    def getDimension(self):
        """
        """
        return self.nlp.getDimension()

    def getFHessian(self, x):
        """
        """
        return run_with_timeout(
            lambda: self.nlp.getFHessian(x),
            seconds=self.timeout)

    def getFeatureTypes(self):
        """
        """
        return self.nlp.getFeatureTypes()

    def getInitializationSample(self):
        """
        """
        return self.nlp.getInitializationSample()

    def report(self, verbose):
        """
        """
        header = "Mathematical Program with timeout\n"
        out = self.nlp.report(verbose)
        return header + out
