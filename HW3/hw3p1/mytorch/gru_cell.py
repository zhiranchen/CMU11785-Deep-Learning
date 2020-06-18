import numpy as np
from activation import *

class GRU_Cell:
    """docstring for GRU_Cell"""
    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t=0

        self.Wzh = np.random.randn(h,h)
        self.Wrh = np.random.randn(h,h)
        self.Wh  = np.random.randn(h,h)

        self.Wzx = np.random.randn(h,d)
        self.Wrx = np.random.randn(h,d)
        self.Wx  = np.random.randn(h,d)

        self.dWzh = np.zeros((h,h))
        self.dWrh = np.zeros((h,h))
        self.dWh  = np.zeros((h,h))

        self.dWzx = np.zeros((h,d))
        self.dWrx = np.zeros((h,d))
        self.dWx  = np.zeros((h,d))

        self.z_act = Sigmoid()
        self.r_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here


    def init_weights(self, Wzh, Wrh, Wh, Wzx, Wrx, Wx):
        self.Wzh = Wzh
        self.Wrh = Wrh
        self.Wh = Wh
        self.Wzx = Wzx
        self.Wrx = Wrx
        self.Wx  = Wx

    def __call__(self, x, h):
        return self.forward(x,h)

    def forward(self, x, h):
        # input:
        #   - x: shape(input dim),  observation at current time-step
        #   - h: shape(hidden dim), hidden-state at previous time-step
        #
        # output:
        #   - h_t: hidden state at current time-step

        self.x = x
        self.hidden = h

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        self.z1 = np.dot(self.Wzh, h)
        self.z2 = np.dot(self.Wzx, x)
        self.z3 = self.z1 + self.z2
        self.z4 = self.z_act(self.z3)
        self.z = self.z4

        self.z5 = np.dot(self.Wrh, h)
        self.z6 = np.dot(self.Wrx, x)
        self.z7 = self.z5 + self.z6
        self.z8 = self.r_act(self.z7)
        self.r = self.z8

        self.z9 = self.z8 * h
        self.z10 = np.dot(self.Wh, self.z9)
        self.z11 = np.dot(self.Wx, x)
        self.z12 = self.z10 + self.z11
        self.z13 = self.h_act(self.z12)
        self.h_tilda = self.z13

        self.z14 = 1 - self.z4
        self.z15 = self.z14 * h
        self.z16 = self.z4 * self.z13
        self.z17 = self.z15 + self.z16
        h_t = self.z17

        assert self.x.shape == (self.d, )
        assert self.hidden.shape == (self.h, )

        assert self.r.shape == (self.h, )
        assert self.z.shape == (self.h, )
        assert self.h_tilda.shape == (self.h, )
        assert h_t.shape == (self.h, )

        return h_t


    # This must calculate the gradients wrt the parameters and return the
    # derivative wrt the inputs, xt and ht, to the cell.
    def backward(self, delta):
        # input:
        #  - delta:  shape (hidden dim), summation of derivative wrt loss from next layer at
        #            the same time-step and derivative wrt loss from same layer at
        #            next time-step
        # output:
        #  - dx: Derivative of loss wrt the input x
        #  - dh: Derivative  of loss wrt the input hidden h

        # 1) Reshape everything you saved in the forward pass.
        # 2) Compute all of the derivatives
        # 3) Know that the autograders the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.
        d16 = delta
        d15 = delta

        d13 = d16 * self.z4
        d4 = d16 * self.z13

        d14 = d15 * self.hidden
        dh = d15 * self.z14

        d4 += -d14

        d12 = d13 * (1 - self.h_act(self.z12) * self.h_act(self.z12)).T

        d10 = d12
        d11 = d12

        self.dWx += np.dot(d11.T, self.x.reshape(1,-1))
        dx_t = np.dot(d11, self.Wx)

        self.dWh += np.dot(d10.T, np.reshape(self.z9, (1, -1)))
        d9 = np.dot(d10, self.Wh)

        d8 = d9 * self.hidden
        dh += d9 * self.r

        d7 = d8 * self.r_act(self.z7) * (1 - self.r_act(self.z7))

        d5 = d7
        d6 = d7

        self.dWrx += np.dot(d6.T, np.reshape(self.x, (1, -1)))
        dx_t += np.dot(d6, self.Wrx)

        self.dWrh += np.dot(d5.T, np.reshape(self.hidden, (1, -1)))
        dh += np.dot(d5, self.Wrh)

        d3 = d4 * self.z_act(self.z3) * (1 - self.z_act(self.z3))

        d2 = d3
        d1 = d3

        self.dWzx += np.dot(d2.T, np.reshape(self.x, (1, -1)))
        dx_t += np.dot(d2, self.Wzx)

        self.dWzh += np.dot(d1.T, np.reshape(self.hidden, (1, -1)))
        dh += np.dot(d1, self.Wzh)


        assert dx_t.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        # return dx, dh
        return dx_t, dh