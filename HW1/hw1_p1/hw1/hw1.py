"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)
        self.layersDim = [input_size] + hiddens + [output_size]
        self.linear_layers = [Linear(inSize, outSize, weight_init_fn, bias_init_fn) \
                              for inSize, outSize in zip(self.layersDim[:-1], self.layersDim[1:])]
        #self.linear_layers = [Linear(input_size, output_size, weight_init_fn, bias_init_fn) for i in range(self.nlayers)]

        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn:
            self.bn_layers = []
            for i in range(self.num_bn_layers):
                self.bn_layers.append(BatchNorm(self.layersDim[i+1]))


    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through your entire MLP.
        input = x
        for i in range(self.nlayers):
            z = self.linear_layers[i].forward(input)
            
            # Batch norm forward
            if self.bn:
                if i < self.num_bn_layers:
                    if self.train_mode:
                        z = self.bn_layers[i].forward(z)
                    else:
                        z = self.bn_layers[i].forward(z, eval=True)

            input = self.activations[i].forward(z)
        return input

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        for i in range(len(self.linear_layers)):
            self.linear_layers[i].dW.fill(0.0)
            self.linear_layers[i].db.fill(0.0)

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)
        if self.momentum:
            for i in range(len(self.linear_layers)):
                # update momentum
                self.linear_layers[i].momentum_W = self.momentum * self.linear_layers[i].momentum_W - self.lr * self.linear_layers[i].dW
                self.linear_layers[i].momentum_B = self.momentum * self.linear_layers[i].momentum_B - self.lr * self.linear_layers[i].db
                # update weights and biases
                self.linear_layers[i].W += self.linear_layers[i].momentum_W 
                self.linear_layers[i].b += self.linear_layers[i].momentum_B
        else:
            for i in range(len(self.linear_layers)):
                # Update weights and biases here
                self.linear_layers[i].W -= (self.lr * self.linear_layers[i].dW)
                self.linear_layers[i].b -= (self.lr * self.linear_layers[i].db)
        # Do the same for batchnorm layers
        if self.bn:
            for i in range(len(self.bn_layers)):
                self.bn_layers[i].gamma -= (self.lr * self.bn_layers[i].dgamma)
                self.bn_layers[i].beta -= (self.lr * self.bn_layers[i].dbeta)


    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.

        # Output
        loss = self.criterion.forward(self.activations[self.nlayers-1].state, labels)
        dl_dz = self.criterion.derivative()

        # Hidden layers
        dz = [] # input of activation
        dy = [] # output of activation
        dnorm = []
        dy.append(dl_dz)
        for i in range(self.nlayers - self.num_bn_layers):
            dz.append(np.multiply(dy[i], self.activations[self.nlayers-i-1].derivative()))
            dy.append(self.linear_layers[self.nlayers-i-1].backward(dz[i]))
        # Batch norm
        for i in range(self.nlayers - self.num_bn_layers, self.nlayers):
            dnorm.append(np.multiply(dy[i], self.activations[self.nlayers-i-1].derivative()))
            dz.append(self.bn_layers[self.nlayers-i-1].backward(dnorm[i-(self.nlayers-self.num_bn_layers)]))
            dy.append(self.linear_layers[self.nlayers-i-1].backward(dz[i]))

    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...
    for e in range(nepochs):
        # Per epoch setup ...
        # shuffle training data
        np.random.shuffle(idxs)
        x_train = trainx[idxs]
        y_train = trainy[idxs]
        mlp.train()
        for b in range(0, len(trainx), batch_size):
            # Train ...
            # 1. Zerofill derivatives after each batch
            mlp.zero_grads()
            # 2. Forward
            y_pred_t = mlp.forward(x_train[b:b+batch_size])
            y_true_t = y_train[b:b+batch_size]
            # 3. Backward
            mlp.backward(y_true_t)
            # 4. Update with gradients
            mlp.step()
            # 5. Calculate training loss
            loss = []
            for element in SoftmaxCrossEntropy().forward(y_pred_t, y_true_t):
                loss.append(element)
            training_losses[e] += sum(loss)
            # 6. Calculate training error count
            for i in range(y_pred_t.shape[0]):
                if np.argmax(y_pred_t[i]) != np.argmax(y_true_t[i]):
                    training_errors[e] += 1
        mlp.eval()
        for b in range(0, len(valx), batch_size):
            # Validate ...
            # 1. Zerofill derivatives after each batch
            mlp.zero_grads()
            # 2. Forward
            y_pred_v = mlp.forward(valx[b:b+batch_size])
            y_true_v = valy[b:b+batch_size]
            # 3. Calculate validation loss
            loss = []
            for element in SoftmaxCrossEntropy().forward(y_pred_v, y_true_v):
                loss.append(element)
            validation_losses[e] += sum(loss)
            # 4. Calculate validation error count
            for i in range(y_pred_v.shape[0]):
                if np.argmax(y_pred_v[i]) != np.argmax(y_true_v[i]):
                    validation_errors[e] += 1


        # Accumulate data...
        training_losses[e] = training_losses[e] / trainx.shape[0]
        validation_losses[e] = validation_losses[e] / valx.shape[0]
        training_errors[e] = training_errors[e] / trainx.shape[0]
        validation_errors[e] = validation_errors[e] / valx.shape[0]

    # Cleanup ...

    # Return results ...

    return (training_losses, training_errors, validation_losses, validation_errors)
