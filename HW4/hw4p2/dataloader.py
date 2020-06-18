import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

LETTER_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']


'''
Optional, create dictionaries for letter2index and index2letter transformations
'''
def create_dictionaries(letter_list):
    n = len(letter_list)
    letter2index = {letter_list[i]:i for i in range(0, n)}
    index2letter = {i:letter_list[i] for i in range(0, n)}
    return letter2index, index2letter

letter2index, index2letter = create_dictionaries(LETTER_LIST)

'''
Loading all the numpy files containing the utterance information and text information
'''
def load_data(dataPath):
    speech_train = np.load(dataPath+"train_new.npy", allow_pickle=True, encoding='bytes')
    speech_dev = np.load(dataPath+"dev_new.npy", allow_pickle=True, encoding='bytes')
    speech_test = np.load(dataPath+"test_new.npy", allow_pickle=True, encoding='bytes')

    transcript_train = np.load(dataPath+"train_transcripts.npy", allow_pickle=True, encoding='bytes')
    transcript_dev = np.load(dataPath+"dev_transcripts.npy", allow_pickle=True, encoding='bytes')

    return speech_train, speech_dev, speech_test, transcript_train, transcript_dev


'''
Transforms alphabetical input to numerical input, replace each letter by its corresponding
index from letter_list
'''
def transform_letter_to_index(transcript):
    '''
    :param transcript :(N, ) Transcripts are the text input
    :param letter_list: Letter list defined above
    :return letter_to_index_list: Returns a list for all the transcript sentence to index
    '''
    letter_to_index_list = []
    for sent in transcript:
        letters = [letter2index['<sos>']]
        for word in sent:
            # Converte from byte format to string for mapping
            s = word.decode('utf-8')
            for c in s:
                letters.append(letter2index[c])
            # Space between each word
            letters.append(letter2index[' '])
        letters.pop()
        letters.append(letter2index['<eos>'])
        letter_to_index_list.append(letters)
    return letter_to_index_list

def transform_index_to_letter(index, stopIdxs):
    index_to_letter_list = []
    for r in index:
        curr = ""
        for i in r:
            # Reached the end of the sentence
            if i in stopIdxs:
                break
            else:
                curr += index2letter[i]
        index_to_letter_list.append(curr)
    return index_to_letter_list


class Speech2TextDataset(Dataset):
    '''
    Dataset class for the speech to text data, this may need some tweaking in the
    getitem method as your implementation in the collate function may be different from
    ours.
    '''
    def __init__(self, speech, text):
        self.dataX = speech
        self.dataY = text

    def __len__(self):
        return self.dataX.shape[0]

    def __getitem__(self, index):
        if self.dataY == None: # test scenario
            return torch.tensor(self.dataX[index].astype(np.float32))
        else:
            return torch.tensor(self.dataX[index].astype(np.float32)), torch.tensor(self.dataY[index])

def collate_train(batch):
    ### Return the padded speech and text data, and the length of utterance and transcript ###
    inputs_pad = []
    targets_pad = []
    inputs_lens = []
    targets_lens = []
    for b in range(len(batch)):
        inputs_pad.append(torch.tensor(batch[b][0]))
        inputs_lens.append(len(batch[b][0]))
        targets_pad.append(torch.tensor(batch[b][1][1:])) # shift one char for target <sos> sentence <eos>
        targets_lens.append(len(batch[b][1])-1) #  sentence <eos>
    inputs_pad = pad_sequence(inputs_pad, batch_first=True) # dim (B, T, C) since batch_first is true, (T, B, C) if false
    targets_pad = pad_sequence(targets_pad, batch_first=True)
    inputs_lens = torch.tensor(inputs_lens)
    targets_lens = torch.tensor(targets_lens)
    return inputs_pad, targets_pad, inputs_lens, targets_lens

def collate_test(batch):
    ### Return padded speech and length of utterance ###
    inputs_pad = []
    inputs_lens = []
    for b in range(len(batch)):
        inputs_pad.append(torch.tensor(batch[b]))
        inputs_lens.append(len(batch[b]))
    inputs_pad = pad_sequence(inputs_pad, batch_first=True)
    inputs_lens = torch.tensor(inputs_lens)
    return inputs_pad, inputs_lens

