# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import time
from torch.utils.data import DataLoader

from dataloader import load_data, transform_letter_to_index, collate_train, collate_test, Speech2TextDataset
from dataloader import LETTER_LIST, letter2index, index2letter
from models import Seq2Seq
from train_test import train, val
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# %%
hyper = {
    'dataPath':"./data/",
    'batchSize':64 if DEVICE=='cuda' else 3,
    'epochs':25,
    'encoder_hidden_dim':256,
    'decoder_hidden_dim':512,
    'embed_dim':256,
    'value_size':128,
    'key_size':128,
    'isAttended':True,
    'displayBatchFreq':50,
    'displayPredFreq':10,
    'checkpointPath':"./checkpoint/",
    "savedCheckpoint": "./checkpoint/init_epoch11.txt",
    'testPredCSVfn':'./data/predicted_test.csv',
    'devPredCSVfn':'./data/predicted_dev.csv',
    'testPredNpyfn':'./data/predicted_test.npy'
}


# %%
# Load datasets
print("*** Load raw data ***")
speech_train, speech_dev, speech_test, transcript_train, transcript_dev = load_data(hyper['dataPath'])


# %%
# Preprocess transcript to char level index
print("*** Process transcript to char level index ***")
character_text_train = transform_letter_to_index(transcript_train)
character_text_dev = transform_letter_to_index(transcript_dev)


# %%
# Get dataloaders
print("*** Get data loaders ***")
train_dataset = Speech2TextDataset(speech_train, character_text_train)
dev_dataset = Speech2TextDataset(speech_dev, character_text_dev)
test_dataset = Speech2TextDataset(speech_test, None)
train_loader = DataLoader(train_dataset, batch_size=hyper['batchSize'], shuffle=True, collate_fn=collate_train) # 387
dev_loader = DataLoader(dev_dataset, batch_size=hyper['batchSize'], shuffle=False, collate_fn=collate_train) # 18
test_loader = DataLoader(test_dataset, batch_size=hyper['batchSize'], shuffle=False, collate_fn=collate_test) # 9


# %%
# Define model and optimizer
print("*** Create the model and define Loss and Optimizer ***")
model = Seq2Seq(input_dim=40, vocab_size=len(LETTER_LIST),                encoder_hidden_dim=hyper['encoder_hidden_dim'],
                decoder_hidden_dim=hyper['decoder_hidden_dim'],
                embed_dim=hyper['embed_dim'],
                value_size=hyper['value_size'],
                key_size=hyper['key_size'],
                isAttended=hyper['isAttended'])
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(reduction='none')
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.75, patience=1, verbose=True, threshold=1e-2)


# %%
model.to(DEVICE)
print(model)


# %%
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

for epoch in range(hyper['epochs']):
    # Print current learnng rate
    for prarm_group in optimizer.param_groups:
        print("Current lr: \t{}".format(prarm_group["lr"]))

    # Train
    print("Start Train \t{} Epoch".format(epoch))
    startTime = time.time()
    train(model, train_loader, criterion, optimizer, epoch, hyper['displayBatchFreq'])
    
    # Save checkpoint
    print("*** Saving Checkpoint ***")
    path = "{}init_epoch{}.txt".format(hyper["checkpointPath"], epoch)
    torch.save({
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)
    print("="*20 + " Epoch {} took {}s".format(epoch, time.time()-startTime) + "="*20)
    
    # Evaluate
    print("Start Dev \t{} Epoch".format(epoch))
    editDist = val(model, dev_loader, criterion, epoch, sampleSize=0, displayBatchFreq=50, displayPredFreq=3)
    scheduler.step(editDist)
    


# %%
checkpoint = torch.load(hyper["savedCheckpoint"])
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)


# %%
# valid inference
validInfer = inference(model, dev_loader, hyper, isValid=True)


# %%
# test inference
testInfer = inference(model, test_loader, hyper, isValid=False)

