# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os

# The following Criterion class will be used again as the basis for a number
# of loss functions (which are in the form of classes so that they can be
# exchanged easily (it's how PyTorch and other ML libraries do it))

class Criterion(object):
    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

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
        self.softmax = None

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        self.labels = y
        x = x - np.amax(x, axis=1).reshape((x.shape[0],1))# Use LogSumExp trick to ensure numerical stability
        self.softmax = np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)
        self.loss = -np.sum(self.labels * np.log(self.softmax), axis=1)
        return self.loss

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """

        return self.softmax - self.labels
