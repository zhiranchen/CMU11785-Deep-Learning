import numpy as np
from activation import *

class RNN_Cell(object):
    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Activation function for
        self.activation = Tanh()

        # hidden dimension and input dimension
        h = self.hidden_size
        d = self.input_size

        # Weights and biases
        self.W_ih = np.random.randn(h, d)
        self.W_hh = np.random.randn(h, h)
        self.b_ih = np.random.randn(h)
        self.b_hh = np.random.randn(h)

        # Gradients
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))

        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    def zero_grad(self):
        d = self.input_size
        h = self.hidden_size
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """
        RNN cell forward (single time step)

        Input (see writeup for explanation)
        ----------
        x : (batch_size, input_size)
        h : (batch_size, hidden_size)

        Returns
        -------
        h_prime : (batch_size, hidden_size)
        """
        h_prime = self.activation(np.dot(x, self.W_ih.T) + self.b_ih + np.dot(h, self.W_hh.T) + self.b_hh)
        return h_prime

    def backward(self, delta, h, h_prev_l, h_prev_t):
        """
        RNN cell backward (single time step)

        Input (see writeup for explanation)
        ----------
        delta : (batch_size, hidden_size)
        h : (batch_size, hidden_size)
        h_prev_l: (batch_size, input_size)
        h_prev_t: (batch_size, hidden_size)

        Returns
        -------
        dx : (batch_size, input_size)
        dh : (batch_size, hidden_size)
        """

        batch_size = delta.shape[0]

        # 0) Done! Step backward through the tanh activation function.
        # Note, because of BPTT, we had to externally save the tanh state, and
        # have modified the tanh activation function to accept an optionally input.
        dz = self.activation.derivative(state=h) * delta # (batch_size, hidden_size)

        # 1) Compute the averaged gradients of the weights and biases
        self.dW_ih += np.dot(dz.T, h_prev_l) / batch_size # (hidden_size, input_size)
        self.dW_hh += np.dot(dz.T, h_prev_t) / batch_size # (hidden_size, hidden_size)
        self.db_ih += np.mean(dz, axis=0) # (hidden_size)
        self.db_hh += np.mean(dz, axis=0) # (hidden_size)

        # 2) Compute dx, dh
        dx = np.dot(dz, self.W_ih) # (batch_size, input_size)
        dh = np.dot(dz, self.W_hh) # (batch_size, hidden_size)

        # 3) Return dx, dh
        return dx, dh
