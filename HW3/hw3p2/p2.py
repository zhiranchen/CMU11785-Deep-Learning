#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Install CTCBeamDecoder Pacakge
get_ipython().system('git clone --recursive https://github.com/parlance/ctcdecode.git')
get_ipython().system('pip install wget')
get_ipython().run_line_magic('cd', 'ctcdecode')
get_ipython().system('pip install .')
get_ipython().run_line_magic('cd', '..')


# In[141]:


# Import packages
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import time
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.append("./hw3p2/")
from phoneme_list import N_STATES, N_PHONEMES, PHONEME_LIST, PHONEME_MAP
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from ctcdecode import CTCBeamDecoder


# In[105]:


# Install Lev Package
get_ipython().system('pip install python-levenshtein')
import Levenshtein as lev


# In[143]:


class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataX = dataset[0]
        self.dataY = dataset[1] if len(dataset) == 2 else None
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.dataX[idx]).float(), torch.from_numpy(self.dataY[idx] + 1 if self.dataY is not None else np.array([-1])).int() # add 1 to label to account for blank
    
    def __len__(self):
        return len(self.dataX)

    
# Model that takes packed sequences in training
class PackedModel(nn.Module):
    def __init__(self, hidden_size, nlayers, out_size=47, embed_size=40):
        super(PackedModel, self).__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.out_size = out_size
        self.cnns = torch.nn.Sequential(
            nn.Conv1d(self.embed_size, self.hidden_size, 3, padding=1, bias=False),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(inplace=True))
        self.rnns = nn.LSTM(input_size=self.hidden_size,
                            hidden_size=self.hidden_size,
                            num_layers=3,
                            bias=True,
                            batch_first=True,
                            dropout=0.2, # regularization
                            bidirectional=True)
        self.hidden2label = torch.nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.Linear(self.hidden_size, self.out_size))
    def forward(self, x, xLens): # x dim (B, T_in, C_in=40)
        x_cnn_input = x.permute(0, 2, 1) # (B, C_in, T_in)
        x_post_cnn = self.cnns(x_cnn_input) # (B, C_out, T_out)
        x_rnn_in = x_post_cnn.permute(2, 0, 1) # (T, B, C_out)
        x_packed = pack_padded_sequence(x_rnn_in, xLens, enforce_sorted=False)
        out_packed, hidden = self.rnns(x_packed)
        out, out_lens = pad_packed_sequence(out_packed, batch_first=True) # (B, T, C)
        
        # Log softmax after output layer is required since nn.CTCLoss expect log prob
        out_prob = self.hidden2label(out).log_softmax(2) # (B, T, Classes=47)
        
        # Permute to fit for input format of CTCLoss
        out_prob = out_prob.permute(1, 0, 2) #torch.transpose(out_prob, 0, 1) # (T, B, C)
        
        # TODO: calculate new xLens
        return out_prob, xLens

    
def getLoaders(train, dev, test, batchSize):
    trainX, trainY = train
    devX, devY = dev
    testX, _ = test
    
    print("*** Create data loader ***")
    # Train
    train_loader_args = dict(shuffle=True, batch_size=batchSize, num_workers=8, collate_fn=pad_collate, pin_memory=True)
    train_loader = DataLoader(MyDataset(train), **train_loader_args)
    
    # Dev
    dev_loader = DataLoader(MyDataset(dev), **train_loader_args)
    
    # Test
    test_loader_args = dict(shuffle=False, batch_size=batchSize, num_workers=8, collate_fn=pad_collate, pin_memory=True)
    test_loader = DataLoader(MyDataset(test), **test_loader_args)
    
    return train_loader, dev_loader, test_loader


def decode(output_probs, dataLens, beamWidth):
    decoder = CTCBeamDecoder(labels=PHONEME_MAP, beam_width=beamWidth,
                            num_processes=os.cpu_count(), log_probs_input=True)
    output_probs = torch.transpose(output_probs, 0, 1) # post transpose: (B, T, C=47)
    output, _, _, out_seq_len = decoder.decode(output_probs, dataLens) # output dim: (BatchSize, Beamwith, T), Out_seq_len dim (batchsize, bewmwidth)
    decodedListShort = []
    decodedListLong = []
    for b in range(output_probs.size(0)):
        currDecode = ""
        if out_seq_len[b][0] != 0:
            currDecodeShort = "".join([PHONEME_MAP[i] for i in output[b, 0, :out_seq_len[b][0]]])
            currDecodeLong = "".join([PHONEME_LIST[i] for i in output[b, 0, :out_seq_len[b][0]]])
        decodedListShort.append(currDecodeShort)
        decodedListLong.append(currDecodeLong)
        
    return decodedListShort, decodedListLong


def idx2phonemes(target):
    return "".join([PHONEME_MAP[x] for x in target])

def calculateLevScore(w1, w2):
    return lev.distance(w1.replace(" ", ""), w2.replace(" ", ""))

def train_epoch(mode, data_loader, criterion, optimizer, epoch):
    model.train()
    start_time = time.time()
    for batch_idx, (data, target, dataLens, targetLens) in enumerate(data_loader):
        optimizer.zero_grad()
        data, target, dataLens, targetLens = data.cuda(), target.cuda(), dataLens.cuda(), targetLens.cuda()

        output, dataLens_new = model(data, dataLens) # out dim: (T, B, C)
        loss = criterion(output, # (T, B, C) T is the largest len in the batch
                         target, # (B, S), S is the largest len in the batch
                         dataLens_new, # (B,), len of sequences in output_log_prob
                         targetLens) # (B,)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print("Epoch: {}\tBatch: {}\tTimestamp: {}".format(epoch, batch_idx, time.time() - start_time))
        
        torch.cuda.empty_cache()
        del data
        del target
        del dataLens
        del targetLens

        
def test_epoch(model, data_loader, epoch, decodeMode=False):
    with torch.no_grad():
        model.eval()
        start_time = time.time()
        running_loss = 0.0
        running_charErr = 0.0
        totalSampleCnt = 0
        
        for batch_idx, (data, target, dataLens, targetLens) in enumerate(data_loader):
            data, target, dataLens, targetLens = data.cuda(), target.cuda(), dataLens.cuda(), targetLens.cuda()
            output, dataLens_new = model(data, dataLens)
            loss = criterion(output,
                             target,
                             dataLens_new,
                             targetLens)
            
            running_loss += loss.item()
            totalSampleCnt += len(data)
            if decodeMode:
                decodedStringsShort, decodedStringsLong = decode(output, dataLens, hyper["beamWidth"])
                targetStrings = [idx2phonemes(i) for i in target]
                for i in range(len(targetStrings)):
                    currCharErr = calculateLevScore(decodedStringsShort[i], targetStrings[i])
                    running_charErr += currCharErr
            if batch_idx % 50 == 0:
                print("Epoch: {}\tBatch: {}\tTimestamp: {}".format(epoch, batch_idx, time.time() - start_time))
            torch.cuda.empty_cache()
            del data
            del target
            del dataLens
            del targetLens
        loss_per_sample = running_loss / len(data_loader)
        dist_per_sample = running_charErr / len(data_loader)
        return loss_per_sample, dist_per_sample
                
def predict(model, data_loader):
    model.eval()
    resShort = np.array([])
    resLong = np.array([])
    start_time = time.time()
    totalSampleCnt = 0
    for batch_idx, (data, target, dataLens, targetLens) in enumerate(data_loader):
        data, target, dataLens, targetLens = data.cuda(), target.cuda(), dataLens.cuda(), targetLens.cuda()
        output, dataLens_new = model(data, dataLens)
        
        decodedStringsShort, decodedStringsLong = decode(output, dataLens, hyper["beamWidth"])
        resShort = np.concatenate((resShort, decodedStringsShort))
        resLong = np.concatenate((resLong, decodedStringsLong))
        print("Predict \tBatch: {}\tTimestamp: {}".format(batch_idx, time.time() - start_time))
        torch.cuda.empty_cache()
        del data
        del target
        del dataLens
        del targetLens
        
    return resShort, resLong


def pad_collate(batch):
    # reference from tutorial: https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
    # sortedBatch = batch # sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    inputs = [x[0] for x in batch]
    targets = [x[1] for x in batch]
    inputs_pad = pad_sequence(inputs, batch_first=True) # dim (B, T, C) since batch_first is true, (T, B, C) if false
    targets_pad = pad_sequence(targets, batch_first=True)
    inputs_lens = torch.LongTensor([len(x) for x in inputs])
    targets_lens = torch.LongTensor([len(x) for x in targets])
    return inputs_pad, targets_pad, inputs_lens, targets_lens


# In[144]:


def main(hyper):
    # Load datasets
    print("*** Load raw data ***")
    train = (np.load(os.path.join(hyper["dataPath"], "wsj0_train"), allow_pickle=True),
            (np.load(os.path.join(hyper["dataPath"], "wsj0_train_merged_labels.npy"), allow_pickle=True)))
    dev = (np.load(os.path.join(hyper["dataPath"], "wsj0_dev.npy"), allow_pickle=True),
            (np.load(os.path.join(hyper["dataPath"], "wsj0_dev_merged_labels.npy"), allow_pickle=True)))
    test = (np.load(os.path.join(hyper["dataPath"], "wsj0_test"), allow_pickle=True), None)
    
    # Get data loaders
    train_loader, dev_loader, test_loader = getLoaders(train, dev, test, hyper["batchSize"])
    
    # Set random seed
    np.random.seed(hyper["seed"])
    torch.manual_seed(hyper["seed"])
    torch.cuda.manual_seed(hyper["seed"])
    
    # Add blank space for phoneme map
    PHONEME_MAP = [" "] + PHONEME_MAP
    PHONEME_LIST = [" "] + PHONEME_LIST
    
    # Create the model and define the Loss an Optimizer
    print("*** Create the model and define Loss and Optimizer ***")
    model = PackedModel(hidden_size=hyper["hiddenSize"], nlayers=hyper["nlayers"], out_size=47, embed_size=40)
    checkpoint = torch.load(hyper["savedCheckpoint"])
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = optim.Adam(model.parameters(), lr=hyper["lr"], weight_decay=hyper["weightDecay"])
    criterion = nn.CTCLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1, verbose=True)
    model.cuda()
    print(model)
    
    # Train the model for N epochs
    for i in range(hyper["nEpochs"]):
        # Print current learnng rate
        for prarm_group in optimizer.param_groups:
            print("Current lr: \t{}".format(prarm_group["lr"]))

        # Trian
        print("Train\tEpoch: {}".format(i))
        startTime = time.time()
        train_epoch(model, train_loader, criterion, optimizer, i)

        # Evaluate
        print("Evaluate Train \tEpoch: {}".format(i))
        train_lossPerSample, train_distPerSample = test_epoch(model, train_loader, i)
        print('Train_LossPerSample: {:.4f}\tTrain_DistPerSample: {:.4f}'.format(
            train_lossPerSample, train_distPerSample))
        print("Evaluate Dev \tEpoch: {}".format(i))
        dev_lossPerSample, dev_distPerSample = test_epoch(model, dev_loader, i)
        print('Dev_LossPerSample: {:.4f}\tDev_DistPerSample: {:.4f}'.format(
            dev_lossPerSample, dev_distPerSample))

        scheduler.step(dev_lossPerSample)

        # Save checkpoint
        print("*** Saving Checkpoint ***")
        path = "{}CNN1_Cont_Epoch{}.txt".format(hyper["checkpointPath"], i)
        torch.save({
            "epoch":i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, path)
        print("="*20 + " Epoch {} took {}s".format(i, time.time()-startTime) + "="*20)
    
    # Predict and save
    resShort, resLong = predict(model, test_loader)
    np.save(hyper["testLabelName"], resShort)
    idxs = np.array(list(range(len(resShort))))
    df = pd.DataFrame({"id" : idxs, "Predicted" : resShort})
    df.to_csv(hyper["testLabelCSVfn"], index=False)
    
    


# In[145]:


if __name__ == "__main__":
    hyper = {
        "dataPath": "./hw3p2",
        "batchSize": 64,
        "lr":5e-4,
        "weightDecay":5e-5,
        "hiddenSize": 256,
        "nlayers":3,
        "nEpochs":20,
        "beamWidth":30,
        "checkpointPath": "./checkpoint/",
        "seed":20,
        "testLabelName" : "./data/predicted.npy",
        "testLabelCSVfn": "./data/predicted.csv",
        "savedCheckpoint": "./checkpoint/CNN1_Cont_Epoch6.txt"
    }
    main(hyper)


# In[ ]:




