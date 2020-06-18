import sys, pdb, os
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from test import Test

sys.path.append('mytorch')
from rnn_cell import *
from loss import *

sys.path.append('hw3')
from rnn_classifier import *

# Reference Pytorch RNN Model
class Reference_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rnn_layers=2):
        super(Reference_Model, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=rnn_layers, bias=True, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x, init_h=None):
        out, hidden = self.rnn(x, init_h)
        out = self.output(out[:,-1,:])
        return out


class RNN_Test(Test):
    def __init__(self):
        pass

    def test_rnncell_forward(self):
        np.random.seed(11785)
        torch.manual_seed(11785)
        # Using i within this loop to vary the inputs
        for i in range(1, 6):

            # Make pytorch rnn cell and get weights
            pytorch_rnn_cell = nn.RNNCell(i*2, i*3)
            state_dict = pytorch_rnn_cell.state_dict()
            W_ih, W_hh = state_dict['weight_ih'].numpy(), state_dict['weight_hh'].numpy()
            b_ih, b_hh = state_dict['bias_ih'].numpy(), state_dict['bias_hh'].numpy()

            # Set user cell and weights
            user_cell = RNN_Cell(i*2, i*3)
            user_cell.init_weights(W_ih, W_hh, b_ih, b_hh)

            # Get inputs
            time_steps = i*2
            inp = torch.randn(time_steps, i*2, i*2)
            hx = torch.randn(i*2, i*3)
            hx_user = hx

            # Loop through inputs
            for t in range(time_steps):
                hx = pytorch_rnn_cell(inp[t], hx)
                hx_user = user_cell(inp[t], hx_user)
                assert(np.allclose(hx.detach().numpy(), hx_user, rtol=1e-03))

        return True

    def test_rnncell_backward(self):
        expected_results = np.load(os.path.join('autograder', 'hw3_autograder',
                               'data', 'rnncell_backward.npy'), allow_pickle = True)
        dx1_, dh1_, dx2_, dh2_, dW_ih_, dW_hh_, db_ih_, db_hh_ = expected_results

        np.random.seed(11785)
        torch.manual_seed(11785)

        batch_size = 3
        input_size = 10
        hidden_size = 20
        user_cell = RNN_Cell(10, 20)

        # Run backward once
        delta = np.random.randn(batch_size, hidden_size)
        h = np.random.randn(batch_size, hidden_size)
        h_prev_l = np.random.randn(batch_size, input_size)
        h_prev_t = np.random.randn(batch_size, hidden_size)
        dx1, dh1 = user_cell.backward(delta, h, h_prev_l, h_prev_t)

        # Run backward again
        delta = np.random.randn(batch_size, hidden_size)
        h = np.random.randn(batch_size, hidden_size)
        h_prev_l = np.random.randn(batch_size, input_size)
        h_prev_t = np.random.randn(batch_size, hidden_size)
        dx2, dh2 = user_cell.backward(delta, h, h_prev_l, h_prev_t)

        dW_ih, dW_hh = user_cell.dW_ih, user_cell.dW_hh
        db_ih, db_hh = user_cell.db_ih, user_cell.db_hh

        # Verify derivatives
        assert(np.allclose(dx1, dx1_, rtol=1e-04))
        assert(np.allclose(dx2, dx2_, rtol=1e-04))
        assert(np.allclose(dh1, dh1_, rtol=1e-04))
        assert(np.allclose(dh2, dh2_, rtol=1e-04))
        assert(np.allclose(dW_ih, dW_ih_, rtol=1e-04))
        assert(np.allclose(dW_hh, dW_hh_, rtol=1e-04))
        assert(np.allclose(db_ih, db_ih_, rtol=1e-04))
        assert(np.allclose(db_hh, db_hh_, rtol=1e-04))

        # Use to save test data for next semester
        # results = [dx1, dh1, dx2, dh2, dW_ih, dW_hh, db_ih, db_hh]
        # np.save(os.path.join('autograder', 'hw3_autograder',
        #                      'data', 'rnncell_backward.npy'), results, allow_pickle=True)
        return True

    def test_rnn_classifier(self):
        rnn_layers = 2
        batch_size = 5
        seq_len = 10
        input_size = 40
        hidden_size = 32 # hidden_size > 100 will cause precision error
        output_size = 138

        np.random.seed(11785)
        torch.manual_seed(11785)

        data_x = np.random.randn(batch_size, seq_len, input_size)
        data_y = np.random.randint(0, output_size, batch_size)

        # Initialize
        # Reference model
        rnn_model = Reference_Model(input_size, hidden_size, output_size, rnn_layers=rnn_layers)
        model_state_dict = rnn_model.state_dict()
        # My model
        my_rnn_model = RNN_Phoneme_Classifier(input_size, hidden_size, output_size, num_layers=rnn_layers)
        rnn_weights = [[model_state_dict['rnn.weight_ih_l%d' % l].numpy(),
                    model_state_dict['rnn.weight_hh_l%d' % l].numpy(),
                    model_state_dict['rnn.bias_ih_l%d' % l].numpy(),
                    model_state_dict['rnn.bias_hh_l%d' % l].numpy()] for l in range(rnn_layers)]
        fc_weights = [model_state_dict['output.weight'].numpy(), model_state_dict['output.bias'].numpy()]
        my_rnn_model.init_weights(rnn_weights, fc_weights)

        # Test forward pass
        # Reference model
        ref_init_h = nn.Parameter(torch.zeros(rnn_layers, batch_size, hidden_size, dtype=torch.float), requires_grad=True)
        ref_out_tensor = rnn_model(torch.FloatTensor(data_x), ref_init_h)
        ref_out = ref_out_tensor.detach().numpy()

        # My model
        my_out = my_rnn_model(data_x)

        # Verify forward outputs
        print('Testing RNN Classifier Forward...')
        assert(np.allclose(my_out, ref_out, rtol=1e-03))
        # if not self.assertions(my_out, ref_out, 'closeness', 'RNN Classifier Forwrd'): #rtol=1e-03)
            # return 'RNN Forward'
        print('RNN Classifier Forward: PASS' )
        print('Testing RNN Classifier Backward...')

        # Test backward pass
        # Reference model
        criterion = nn.CrossEntropyLoss()
        loss = criterion(ref_out_tensor, torch.LongTensor(data_y))
        ref_loss = loss.detach().item()
        rnn_model.zero_grad()
        loss.backward()
        grad_dict = {k:v.grad for k, v in zip(rnn_model.state_dict(), rnn_model.parameters())}
        dh = ref_init_h.grad

        # My model
        my_criterion = SoftmaxCrossEntropy()
        my_labels_onehot = np.zeros((batch_size, output_size))
        my_labels_onehot[np.arange(batch_size), data_y] = 1.0
        my_loss = my_criterion(my_out, my_labels_onehot).mean()
        delta = my_criterion.derivative()
        my_dh = my_rnn_model.backward(delta)

        # Verify derivative w.r.t. each network parameters
        assert(np.allclose(my_dh, dh.detach().numpy(), rtol=1e-04))
        assert(np.allclose(my_rnn_model.output_layer.dW, grad_dict['output.weight'].detach().numpy(), rtol=1e-03))
        assert(np.allclose(my_rnn_model.output_layer.db, grad_dict['output.bias'].detach().numpy()))
        for l, rnn_cell in enumerate(my_rnn_model.rnn):
            assert(np.allclose(my_rnn_model.rnn[l].dW_ih, grad_dict['rnn.weight_ih_l%d' % l].detach().numpy(), rtol=1e-03))
            assert(np.allclose(my_rnn_model.rnn[l].dW_hh, grad_dict['rnn.weight_hh_l%d' % l].detach().numpy(), rtol=1e-03))
            assert(np.allclose(my_rnn_model.rnn[l].db_ih, grad_dict['rnn.bias_ih_l%d' % l].detach().numpy(), rtol=1e-03))
            assert(np.allclose(my_rnn_model.rnn[l].db_hh, grad_dict['rnn.bias_hh_l%d' % l].detach().numpy(), rtol=1e-03))

        print('RNN Classifier Backward: PASS' )
        return True



    def run_test(self):
        # Test forward
        self.print_name('Section 3.1 - RNN Forward')
        forward_outcome = self.test_rnncell_forward()
        self.print_outcome('RNN Forward', forward_outcome)
        if forward_outcome == False:
            self.print_failure('RNN Forward')
            return False

        # Test Backward
        self.print_name('Section 3.2 - RNN Backward')
        backward_outcome = self.test_rnncell_backward()
        self.print_outcome('RNN backward', backward_outcome)
        if backward_outcome == False:
            self.print_failure('RNN Backward')
            return False

        # Test RNN Classifier
        self.print_name('Section 3.3 - RNN Classifier')
        classifier_outcome = self.test_rnn_classifier()
        self.print_outcome('RNN Classifier', classifier_outcome)
        if classifier_outcome == False:
            self.print_failure(classifier_outcome)
            return False

        return True
