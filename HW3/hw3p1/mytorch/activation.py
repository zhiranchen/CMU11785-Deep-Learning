# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import numpy as np

class Activation(object):

    """
    Interface for activation functions (non-linearities).
    """

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class Sigmoid(Activation):

    """
    Sigmoid activation function
    """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        self.state = (1 / (1 + np.exp(-x)))
        return self.state

    def derivative(self):
        return (self.state) * (1 - self.state)


class Tanh(Activation):

    """
    Modified Tanh to work with BPTT.
    The tanh(x) result has to be stored elsewhere otherwise we will
    have to store results for multiple timesteps in this class for each cell,
    which could be considered bad design.

    Now in the derivative case, we can pass in the stored hidden state and
    compute the derivative for that state instead of the "current" stored state
    which could be anything.
    """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = np.tanh(x)
        return self.state

    def derivative(self, state=None):
        if state is not None:
            return 1 - (state**2)
        else:
            return 1 - (self.state**2)

