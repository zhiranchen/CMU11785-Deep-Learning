import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import random
from dataloader import letter2index
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns
import time


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Attention(nn.Module):
    '''
    Attention is calculated using key, value and query from Encoder and decoder.
    Below are the set of operations you need to perform for computing attention:
        energy = bmm(key, query)
        attention = softmax(energy)
        context = bmm(attention, value)
    '''
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, lens):
        '''
        :param query :(N, context_size) Query is the output of LSTMCell from Decoder
        :param key: (N, T_max, key_size) Key Projection from Encoder per time step
        :param value: (N, T_max, value_size) Value Projection from Encoder per time step
        :param lens: (N, T) Length of key and value, used for binary masking
        :return output: Attended Context
        :return attention: Attention mask that can be plotted
        '''
#         print("key.size {}".format(key.size()))
#         print("query.size {}".format(query.size()))
        energy = torch.bmm(key, query.unsqueeze(2)).squeeze(2) # (N, T_max, key_size) * (N, context_size, 1) = (N, T_max, 1) -> (N, T_max)
#         print("enery.size {}".format(energy.size()))

        # binary masking for padded positions
        mask = torch.arange(key.size(1)).unsqueeze(0) >= lens.unsqueeze(1) # (1, T) >= (B, 1) -> (N, T_max)
#         print("mask.size {}".format(mask.size()))
        mask = mask.to(DEVICE)
        energy.masked_fill_(mask, -1e9) # (N, T_max)
        attention = nn.functional.softmax(energy, dim=1) # (N, T_max)
        output = torch.bmm(attention.unsqueeze(1), value).squeeze(1) # (N, T_max)

        return output, attention


class pBLSTM(nn.Module):
    '''
    Pyramidal BiLSTM
    The length of utterance (speech input) can be hundereds to thousands of frames long.
    The Paper reports that a direct LSTM implementation as Encoder resulted in slow convergence,
    and inferior results even after extensive training.
    The major reason is inability of AttendAndSpell operation to extract relevant information
    from a large number of input steps.
    '''
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, x):
        '''
        :param x :(N, T) input to the pBLSTM
        :return output: (N, T, H) encoded sequence from pyramidal Bi-LSTM
        '''

        x_padded, x_lens = pad_packed_sequence(x, batch_first=True)
        x_lens = x_lens.to(DEVICE)

        # chop off extra odd/even sequence
        x_padded = x_padded[:, :(x_padded.size(1) // 2) * 2, :] # (B, T, dim)

        # reshape to (B, T/2, dim*2)
        x_reshaped = x_padded.reshape(x_padded.size(0), x_padded.size(1) // 2, x_padded.size(2) * 2)
        x_lens = x_lens // 2

        x_packed = pack_padded_sequence(x_reshaped, lengths=x_lens, batch_first=True, enforce_sorted=False)


        out, _ = self.blstm(x_packed)
        return out




class Encoder(nn.Module):
    '''
    Encoder takes the utterances as inputs and returns the key and value.
    Key and value are nothing but simple projections of the output from pBLSTM network.
    '''
    def __init__(self, input_dim, hidden_dim, value_size=128,key_size=128):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

        ### Add code to define the blocks of pBLSTMs! ###
        self.pBLSTMs = nn.Sequential(
            pBLSTM(hidden_dim*4, hidden_dim),
            pBLSTM(hidden_dim*4, hidden_dim),
            pBLSTM(hidden_dim*4, hidden_dim)
        )

        self.key_network = nn.Linear(hidden_dim*2, value_size)
        self.value_network = nn.Linear(hidden_dim*2, key_size)

    def forward(self, x, lens):
        rnn_inp = pack_padded_sequence(x, lengths=lens, batch_first=True, enforce_sorted=False)

        outputs, _ = self.lstm(rnn_inp)


        ### Use the outputs and pass it through the pBLSTM blocks! ###
        outputs = self.pBLSTMs(outputs)

        linear_input, encoder_lens = pad_packed_sequence(outputs, batch_first=True)
        keys = self.key_network(linear_input)
        value = self.value_network(linear_input)
        return keys, value, encoder_lens


class Decoder(nn.Module):
    '''
    As mentioned in a previous recitation, each forward call of decoder deals with just one time step,
    thus we use LSTMCell instead of LSLTM here.
    The output from the second LSTMCell can be used as query here for attention module.
    In place of value that we get from the attention, this can be replace by context we get from the attention.
    Methods like Gumble noise and teacher forcing can also be incorporated for improving the performance.
    '''
    def __init__(self, vocab_size, decoder_hidden_dim, embed_dim, value_size=128, key_size=128, isAttended=False):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm1 = nn.LSTMCell(input_size=embed_dim + value_size, hidden_size=decoder_hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=decoder_hidden_dim, hidden_size=key_size)

        self.isAttended = isAttended
        if (isAttended == True):
            self.attention = Attention()

        self.character_prob = nn.Linear(key_size + value_size, vocab_size)
        self.value_size = value_size
        self.hidden_dim = decoder_hidden_dim

    def forward(self, key, values, encoder_lens, batch_idx, text=None, isTrain=True, teacherForcingRate = 0.1, isGumbel=False):
        '''
        :param key :(N, T, key_size) Output of the Encoder Key projection layer
        :param values: (N, T, value_size) Output of the Encoder Value projection layer
        :param text: (N, text_len) Batch input of text with text_length
        :param isTrain: Train or eval mode
        :return predictions: Returns the character perdiction probability
        '''
        batch_size = key.shape[0]

        if (isTrain == True):
            max_len =  text.shape[1]
            embeddings = self.embedding(text)
        else:
            max_len = 250

        predictions = []
        hidden_states = [None, None]
        prediction = torch.zeros(batch_size, 1).to(DEVICE)
        context = values[:, 0, :] # initialize context
        attentionPlot = []
        for i in range(max_len):
            # * Implement Gumble noise and teacher forcing techniques
            # * When attention is True, replace values[i,:,:] with the context you get from attention.
            # * If you haven't implemented attention yet, then you may want to check the index and break
            #   out of the loop so you do you do not get index out of range errors.

            if (isTrain):
                # Teacher forcing
                teacher_forcing = True if random.random() > teacherForcingRate else False # currently 0.2
                if not teacher_forcing:
                    # Use previous prediction/initial zeroed prediction for teacher forcing
                    if i != 0 and isGumbel: # use Gumbel noise to add noise to add variety to phoneme
                        char_embed = torch.nn.functional.gumbel_softmax(prediction).mm(self.embedding.weight)
                    else:
                        char_embed = self.embedding(prediction.argmax(dim=-1))
                else:
                    if i == 0:
                        start_char = torch.zeros(batch_size, dtype=torch.long).fill_(letter2index['<sos>']).to(DEVICE)
                        char_embed = self.embedding(start_char)
                    else:
                        # Use ground truth
                        char_embed = embeddings[:, i-1, :]
            else:
                if i == 0:
                    start_char = torch.zeros(batch_size, dtype=torch.long).fill_(letter2index['<sos>']).to(DEVICE)
                    char_embed = self.embedding(start_char)
                else:
                    char_embed = self.embedding(prediction.argmax(dim=-1))

            # Input to decoder is the concatenated char embedding and attention context vector
            inp = torch.cat([char_embed, context], dim=1)
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1]) # output (h_1, c_1)

            ### Compute attention from the output of the second LSTM Cell ###
            output = hidden_states[1][0]

            # Attention plot during training
            if self.isAttended:
                context, attention = self.attention(output, key, values, encoder_lens)
                # plot random sample from batch T * key_size
                if batch_idx % 64 == 0 and isTrain:
                    currAtten = attention[0].detach().cpu()

                    attentionPlot.append(currAtten) #(len of input seq, len of output seq)
            else:
                context = values[:, i, :] if i < values.size(1) else torch.zeros(batch_size, self.value_size).to(DEVICE)

            prediction = self.character_prob(torch.cat([output, context], dim=1))
            predictions.append(prediction.unsqueeze(1))

        # Plot attention plot
        if batch_idx % 64 == 0 and isTrain:
            attentions = torch.stack(attentionPlot, dim=1)

            plt.clf()
            sns.heatmap(attentions, cmap='GnBu')
            plt.savefig("./attention/heat_{}s.png".format(time.time()))



        return torch.cat(predictions, dim=1)


class Seq2Seq(nn.Module):
    '''
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    This is simply a wrapper "model" for your encoder and decoder.
    '''
    def __init__(self, input_dim, vocab_size, encoder_hidden_dim=256, decoder_hidden_dim=512, embed_dim=256, value_size=128, key_size=128, isAttended=False):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, encoder_hidden_dim) #encoder_hidden_dim)
        self.decoder = Decoder(vocab_size, decoder_hidden_dim, embed_dim, value_size, key_size, isAttended)

    def forward(self, speech_input, speech_len, batchNum, text_input=None, isTrain=True):
        key, value, encoder_lens = self.encoder(speech_input, speech_len)
        if (isTrain == True):
            predictions = self.decoder(key, value, encoder_lens, batchNum, text_input)
        else:
            predictions = self.decoder(key, value, encoder_lens, batchNum, text=None, isTrain=False)
        return predictions
