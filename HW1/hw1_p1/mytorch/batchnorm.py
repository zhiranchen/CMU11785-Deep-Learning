# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)
        """
        self.x = x

        if eval: # use running mean and var for testing
            self.norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        else:
            self.mean = np.mean(x, axis=0)
            self.var = np.var(x, axis=0)
            self.norm =  (x - self.mean) / np.sqrt(self.var + self.eps)

            # Update running batch statistics
            self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
            self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var

        self.out = self.gamma * self.norm + self.beta
        return self.out


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        diffX = self.x - self.mean
        dnorm = delta * self.gamma
        self.dbeta = np.sum(delta, axis=0, keepdims=True)
        self.dgamma = np.sum(delta*self.norm, axis=0, keepdims=True)

        sqrtVar = np.sqrt(self.var + self.eps)
        m = delta.shape[0]
        dvar = - np.sum(dnorm * (diffX/(2*(sqrtVar**3))), axis=0)
        dmu = - np.sum(dnorm / sqrtVar, axis=0) - (2 / m) * dvar * np.sum(diffX, axis=0)
        dx = (dnorm/sqrtVar) + (dvar * (2/m) * diffX) + (dmu / m)

        return dx
