import numpy as np

class Linear(object):
    def __init__(self, in_feature, out_feature):
        self.in_feature = in_feature
        self.out_feature = out_feature

        self.W = np.random.randn(out_feature, in_feature)
        self.b = np.zeros(out_feature)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        self.x = x
        self.out = x.dot(self.W.T) + self.b
        return self.out

    def backward(self, delta):
        batch_size = delta.shape[0]
        self.db = np.sum(delta, axis=0) / batch_size
        self.dW = np.dot(delta.T, self.x) / batch_size
        dx = np.dot(delta, self.W)
        return dx

