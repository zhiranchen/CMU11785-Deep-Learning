import numpy as np

class Criterion(object):
    """
    Interface for loss functions.
    """

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):

    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    def forward(self, x, y):
        self.logits = x
        self.labels = y
        exps = np.exp(x - np.max(x, axis=1)[:,None])
        self.sm = exps / np.sum(exps, axis=1)[:,None]
        loss = -np.log((self.sm*y).sum(axis=1))
        return loss

    def derivative(self):
        return self.sm - self.labels
