import numpy as np
from layers import *

# This code is only for your reference for Sections 3.3 and 3.4

class MLP():
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            in_size, out_size = layer_sizes[i], layer_sizes[i+1]
            self.layers.append(Linear(in_size, out_size))
            self.layers.append(ReLU())
        self.layers = self.layers[:-1] # remove final ReLU

    def __call__(self, x):
        return self.forward(x)

    def init_weights(self,  weights):
        for i in range(len(weights)):
            self.layers[i*2].W = weights[i].T

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta

        
if __name__ == '__main__':
    D = 24 # length of each feature vector
    layer_sizes = [8 * D, 8, 16, 4]
    mlp = MLP([8 * D, 8, 16,  4])
