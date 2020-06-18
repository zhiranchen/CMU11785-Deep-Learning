# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

        self.x = None
        self.input_size = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        batch_size, in_channel, input_size = x.shape
        output_size = (input_size - self.kernel_size) // self.stride + 1
        res = np.zeros([batch_size, self.out_channel, output_size])
        self.x = x
        self.input_size = input_size
        for batch in range(res.shape[0]):
            for cOut in range(self.out_channel):
                for i in range(output_size):
                    res[batch, cOut, i] = \
                        np.multiply(x[batch, :, i*self.stride:i*self.stride+self.kernel_size], self.W[cOut, :, :]).sum()
                res[batch, cOut] += self.b[cOut]
        return res
        # startIdx = 0
        # for i in range(output_size):
        #     curr = np.tensordot(x[:, :, startIdx:startIdx + self.kernel_size], self.W, axes=([1,2], [1, 2]))
        #     res.append(curr)
        #     startIdx += self.stride
        # res = np.array(res)
        # print(res.shape)
        # return np.array(res)


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """

        # Calculate dW
        batch_size, out_channel, output_size = delta.shape
        for batch in range(batch_size):
            for cOut in range(self.out_channel):
                for cIn in range(self.in_channel):
                    for i in range(self.kernel_size):
                        for out in range(output_size):
                            self.dW[cOut, cIn, i] += self.x[batch, cIn, i + self.stride * out] * delta[batch, cOut, out]

        # Calculate db
        self.db = np.sum(delta, axis=(0, 2))

        # Calculate dX
        dX = np.zeros(self.x.shape)
        for batch in range(batch_size):
            for cIn in range(self.in_channel):
                for cOut in range(self.out_channel):
                    for s in range((self.input_size - self.kernel_size)//self.stride + 1):
                        for k in range(self.kernel_size):
                            dX[batch, cIn, self.stride * s + k] += delta[batch, cOut, s] * self.W[cOut, cIn, k]

        return dX



class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        return x.reshape(self.b, self.c * self.w)

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        # Return the derivative of the loss with respect to the flatten
        # layer input
        # Calculate dX
        dx = np.reshape(delta, (self.b, self.c, self.w))
        return dx



