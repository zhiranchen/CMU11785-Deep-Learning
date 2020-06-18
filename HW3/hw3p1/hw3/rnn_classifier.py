import numpy as np
import sys

sys.path.append('mytorch')
from rnn_cell import *
from linear import *

# RNN Phoneme Classifier
class RNN_Phoneme_Classifier(object):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        ## TODO: Understand then uncomment this code :)
        self.rnn = [RNN_Cell(input_size, hidden_size) if i == 0 else
                    RNN_Cell(hidden_size, hidden_size) for i in range(num_layers)]
        self.output_layer = Linear(hidden_size, output_size)

        # store hidden states at each time step, [(seq_len+1) * (num_layers, batch_size, hidden_size)]
        self.hiddens = []

    def init_weights(self, rnn_weights, linear_weights):
        """
        Initialize weights

        Parameters
        ----------
        rnn_weights:
        [[W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0],
         [W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1], ...]

        linear_weights:
        [W, b]
        """
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i])
        self.output_layer.init_weights(*linear_weights)

    def __call__(self, x, h_0=None):
        return self.forward(x, h_0)

    def forward(self, x, h_0=None):

        """
        RNN forward, multiple layers, multiple time steps

        Parameters
        ----------
        x : (batch_size, seq_len, input_size)
            Input
        h_0 : (num_layers, batch_size, hidden_size)
            Initial hidden states. Defaults to zeros if not specified

        Returns
        -------
        logits : (batch_size, output_size)
        Output logits
        """

        # Get the batch size and sequence length, and initialize the hidden
        # vectors given the paramters.
        batch_size, seq_len = x.shape[0], x.shape[1]
        if h_0 is None:
            hidden = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        else:
            hidden = h_0

        # Save x and append the hidden vector to the hiddens list
        self.x = x
        self.hiddens.append(hidden.copy())

        ### Add your code here --->
        # (More specific pseudocode may exist in lecture slides): Lecture 13 PPT slide 72
        # Iterate through the sequence
        for t in range(seq_len):
            # Iterate over the length of your self.rnn (through the layers)
            xInput = self.x[:, t, :]
            hidden = []
            for i in range(len(self.rnn)):
                # Run the rnn cell with the correct parameters and update
                # the parameters as needed. Update hidden.
                h_ti = self.rnn[i].forward(xInput, self.hiddens[-1][i])
                xInput = h_ti
                hidden.append(h_ti)
            # Similar to above, append a copy of the current hidden array to the hiddens list
            self.hiddens.append(hidden.copy())

        # Get the outputs from the last time step using the linear layer and return it
        logits = self.output_layer.forward(xInput)
        return logits

    def backward(self, delta):

        """
        RNN Back Propagation Through Time (BPTT)

        Parameters
        ----------
        delta : (batch_size, hidden_size)
        gradient w.r.t. the last time step output dY(seq_len-1)

        Returns
        -------
        dh_0 : (num_layers, batch_size, hidden_size)
        gradient w.r.t. the initial hidden states
        """

        # Initilizations
        batch_size, seq_len = self.x.shape[0], self.x.shape[1]
        dh = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        dh[-1] = self.output_layer.backward(delta)

        '''
        '''
        # Notes:
        # More specific pseudocode may exist in lecture slides and a visualization
        #  exists in the writeup. Lecture 13 PPT slide 94
        # WATCH out for off by 1 errors due to implementation decisions.
        #
        # Pseudocode:
        # Iterate in reverse order of time (from seq_len-1 to 0)
        for t in range(seq_len-1, -1, -1):
            # Iterate in reverse order of layers (from num_layers-1 to 0)
            for i in range(self.num_layers-1, -1, -1):
                # Get h_prev_l either from hiddens or x depending on the layer
                #   (Recall that hiddens has an extra initial hidden state)
                h_prev_l = self.hiddens[t+1][i-1] if i != 0 else self.x[:, t, :]
                h_prev_t = self.hiddens[t][i]
                # Use dh and hiddens to get the other parameters for the backward method
                #   (Recall that hiddens has an extra initial hidden state)
                retdx, retdh = self.rnn[i].backward(dh[i], self.hiddens[t+1][i], h_prev_l, h_prev_t)
                # Update dh with the new dh from the backward pass of the rnn cell
                dh[i] = retdh
                # If you aren't at the first layer, you will want to add dx to
                #   the gradient from l-1th layer
                if i != 0:
                    dh[i-1] += retdx
        # Normalize dh by batch_size since initial hidden states are also treated
        #   as parameters of the network (divide by batch size)
        dh = dh / batch_size

        dh_0 = dh
        return dh_0
