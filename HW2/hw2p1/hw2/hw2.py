# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)​

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from linear import *
from conv import *

class CNN(object):

    """
    A simple convolutional neural network

    Here you build implement the same architecture described in Section 3.3
    You need to specify the detailed architecture in function "get_cnn_model" below
    The returned model architecture should be same as in Section 3.3 Figure 3
    """

    def __init__(self, input_width, num_input_channels, num_channels, kernel_sizes, strides,
                 num_linear_neurons, activations, conv_weight_init_fn, bias_init_fn,
                 linear_weight_init_fn, criterion, lr):
        """
        input_width           : int    : The width of the input to the first convolutional layer
        num_input_channels    : int    : Number of channels for the input layer
        num_channels          : [int]  : List containing number of (output) channels for each conv layer
        kernel_sizes          : [int]  : List containing kernel width for each conv layer
        strides               : [int]  : List containing stride size for each conv layer
        num_linear_neurons    : int    : Number of neurons in the linear layer
        activations           : [obj]  : List of objects corresponding to the activation fn for each conv layer
        conv_weight_init_fn   : fn     : Function to init each conv layers weights
        bias_init_fn          : fn     : Function to initialize each conv layers AND the linear layers bias to 0
        linear_weight_init_fn : fn     : Function to initialize the linear layers weights
        criterion             : obj    : Object to the criterion (SoftMaxCrossEntropy) to be used
        lr                    : float  : The learning rate for the class

        You can be sure that len(activations) == len(num_channels) == len(kernel_sizes) == len(strides)
        """

        # Don't change this -->
        self.train_mode = True
        self.nlayers = len(num_channels)

        self.activations = activations
        self.criterion = criterion

        self.lr = lr
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        ## Your code goes here -->
        # self.convolutional_layers (list Conv1D) = []
        # self.flatten              (Flatten)     = Flatten()
        # self.linear_layer         (Linear)      = Linear(???)
        # <---------------------
        outChannel = num_input_channels
        outSize = 0
        inputSize = input_width
        self.convolutional_layers = []
        for i in range(self.nlayers):
            self.convolutional_layers.append(Conv1D(in_channel=outChannel, out_channel=num_channels[i], \
                                                    kernel_size=kernel_sizes[i], stride=strides[i], \
                                                    weight_init_fn=conv_weight_init_fn, bias_init_fn=bias_init_fn))
            outChannel = num_channels[i]
            outSize = (inputSize - kernel_sizes[i]) // strides[i] + 1
            inputSize = outSize
        self.flatten = Flatten()
        self.linear_layer = Linear(in_feature=outChannel*outSize, out_feature=num_linear_neurons, \
                                   weight_init_fn=linear_weight_init_fn, bias_init_fn=bias_init_fn)


    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, num_input_channels, input_width)
        Return:
            out (np.array): (batch_size, num_linear_neurons)
        """

        ## Your code goes here -->
        # Iterate through each layer
        # <---------------------

        # Save output (necessary for error and loss)
        # self.output = x
        input = x
        for i in range(self.nlayers):
            z = self.convolutional_layers[i].forward(input)
            input = self.activations[i].forward(z)
        input = self.flatten.forward(input)
        self.output = self.linear_layer.forward(input)
        return self.output

    def backward(self, labels):
        """
        Argument:
            labels (np.array): (batch_size, num_linear_neurons)
        Return:
            grad (np.array): (batch size, num_input_channels, input_width)
        """

        m, _ = labels.shape
        self.loss = self.criterion(self.output, labels).sum()
        grad = self.criterion.derivative()

        ## Your code goes here -->
        # Iterate through each layer in reverse order
        # <---------------------
        dy = self.linear_layer.backward(grad) # backprop on linear layer
        dy = self.flatten.backward(dy) # output of activation
        dz = 0 # input of activation
        for i in range(self.nlayers-1, -1, -1):
            dz = np.multiply(dy, self.activations[i].derivative())
            dy = self.convolutional_layers[i].backward(dz)
        grad = dy
        return grad


    def zero_grads(self):
        # Do not modify this method
        for i in range(self.nlayers):
            self.convolutional_layers[i].dW.fill(0.0)
            self.convolutional_layers[i].db.fill(0.0)

        self.linear_layer.dW.fill(0.0)
        self.linear_layer.db.fill(0.0)

    def step(self):
        # Do not modify this method
        for i in range(self.nlayers):
            self.convolutional_layers[i].W = (self.convolutional_layers[i].W -
                                              self.lr * self.convolutional_layers[i].dW)
            self.convolutional_layers[i].b = (self.convolutional_layers[i].b -
                                  self.lr * self.convolutional_layers[i].db)

        self.linear_layer.W = (self.linear_layer.W - self.lr * self.linear_layers.dW)
        self.linear_layers.b = (self.linear_layers.b -  self.lr * self.linear_layers.db)


    def __call__(self, x):
        # Do not modify this method
        return self.forward(x)

    def train(self):
        # Do not modify this method
        self.train_mode = True

    def eval(self):
        # Do not modify this method
        self.train_mode = False
