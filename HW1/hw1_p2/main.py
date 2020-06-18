# Import all the necessary libraries
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pandas as pd

from torch.utils.data import DataLoader, Dataset, TensorDataset

import time
import matplotlib.pyplot as plt

class MyDataset(Dataset):
    # Create own Dataset and append features only when accessed
    def __init__(self, dataset, k):
        self.k = k
        self.dataX = dataset[0]
        self.dataY = dataset[1] if len(dataset) == 2 else None
        self.idxMap = []
        for i, utter in enumerate(self.dataX):
            for j in range(utter.shape[0]):
                self.idxMap.append((i, j)) # frame index, each frame has dim 40
        
    def __getitem__(self, idx):
        i, j = self.idxMap[idx]
        withContext = self.dataX[i].take(range(j - self.k, j + self.k + 1), mode='clip', axis=0).flatten()
        x = torch.Tensor(withContext).float()
        y = self.dataY[i][j] if self.dataY is not None else -1
        return x, y
    
    def __len__(self):
        return len(self.idxMap)

def getLoaders(train, dev, test, datapath, cuda, k, batchSize):
    trainX, trainY = train # 24500 * l *  40 where l could be any length
    devX, devY = dev # 1100 * l * 40
    testX, _ = test # 361 * l * 40
    
    print('*** Create data loader ***')
    # Train
    train_loader_args = dict(shuffle=True, batch_size=batchSize, num_workers=8, pin_memory=True)
    train_loader = DataLoader(MyDataset(train, k), **train_loader_args)

    # Dev
    dev_loader_args = dict(shuffle=True, batch_size=batchSize, num_workers=1, pin_memory=True)
    dev_loader = DataLoader(MyDataset(dev, k), **dev_loader_args)
    
    # Test
    test_loader_args = dict(shuffle=False, batch_size=batchSize, num_workers=1, pin_memory=True)
    test_loader = DataLoader(MyDataset(test, k), **test_loader_args)
    
    return train_loader, dev_loader, test_loader


class MLP(nn.Module):
    def __init__(self, sizeList):
        super(MLP, self).__init__()
        layers = []
        self.sizeList = sizeList
        for i in range(len(sizeList) - 2):
            if i < 5: # batchnorm
                layers.append(nn.Linear(sizeList[i], sizeList[i+1]))
                layers.append(nn.BatchNorm1d(sizeList[i+1]))
                layers.append(nn.LeakyReLU())
            else: # regular layer
                layers.append(nn.Linear(sizeList[i], sizeList[i+1]))
                layers.append(nn.LeakyReLU())
        # Last layer
        layers.append(nn.Linear(sizeList[-2], sizeList[-1]))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


# A function that will train the network for one epoch
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.cuda()
        target = target.cuda()
        
        outputs = model(data)
        loss = criterion(outputs, target)
        running_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        if batch_idx % 1000 == 0:
            print("Finished " + str(batch_idx) + "\t Timestamp: "+ str(time.time() - start_time))

    end_time = time.time()
    running_loss = running_loss / len(train_loader)
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    return running_loss

# A function that will evaluate out network's performance on the test set
def test_model(model, test_loader, criterion, device):
    with torch.no_grad():
        model.eval()
        start_time = time.time()
        
        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.cuda()
            target = target.cuda()
            
            outputs = model(data)
            
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()
            
            loss = criterion(outputs, target).detach()
            running_loss += loss.item()
            if batch_idx % 1000 == 0:
                print("Finished " + str(batch_idx) + "\t Timestamp: "+ str(time.time() - start_time))
        
        running_loss /= len(test_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('Testing Loss: ', running_loss)
        print('Testing Accuracy: ', acc, '%')
        return running_loss, acc


# A function that predicts test set label
def predictLabels(model, test_loader, device):
    model.eval()
    
    res = np.array([])
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.cuda()
        target = target.cuda()
        
        outputs = model(data)
        _, predicted = torch.max(outputs.data, dim=1)
        res = np.concatenate((res, predicted.cpu().numpy().reshape(-1)))
    return res


def main(hyper):
    datapath = hyper['dataPath']
    weightDirName = hyper["weightDirName"]
    cuda = torch.cuda.is_available()
    num_workers = 8 if cuda else 0 
    nEpochs = hyper["nEpochs"]
    
    print('*** Load raw data ***')
    train = (np.load(os.path.join(datapath, 'train.npy'), allow_pickle=True), 
        np.load(os.path.join(datapath, 'train_labels.npy'), allow_pickle=True))
    dev = (np.load(os.path.join(datapath, 'dev.npy'), allow_pickle=True),
        np.load(os.path.join(datapath, 'dev_labels.npy'), allow_pickle=True))
    test = (np.load(os.path.join(datapath, 'test.npy'), allow_pickle=True), None)

    # Get data loaders
    train_loader, dev_loader, test_loader = getLoaders(train, dev, test, datapath, cuda, hyper["kContext"], hyper['batchSize'])
    
    # Create the model and define the Loss and Optimizer
    print("*** Create the model and define Loss and Optimizer ***")
    inputSize = (2 * hyper["kContext"] + 1) * 40 # new dim
    outputSize = 138 # possible phoneme states
    model = MLP([inputSize] + hyper["hiddenDims"] + [outputSize])
    checkpoint = torch.load(hyper["checkpointPath"])
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyper["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)
    device = torch.device("cuda" if cuda else "cpu")
    model.cuda()
    print(model)
    
    # Train the model for N epochs
    print("*** Train the Model for N Epochs ***")
    Train_loss = []
    Test_loss = []
    Test_acc = []
    for i in range(nEpochs):
        print("Train "+ str(i)+" epoch")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print("Dev "+ str(i)+" epoch")
        dev_loss, dev_acc = test_model(model, dev_loader, criterion, device)
        scheduler.step(dev_loss)
        Train_loss.append(train_loss)
        Test_loss.append(dev_loss)
        Test_acc.append(dev_acc)
        print('='*20)
        print("*** Saving Checkpoint ***")
        path = "{}optimContDecreaseCont_Epoch_{}contextK_v2.txt".format(weightDirName, str(i), str(hyper["kContext"]))
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'dev_loss':dev_loss,
            'dev_acc': dev_acc
        }, path)


    # Visualize Training and Validation data
    print("*** Visualize Training and Validation Data ***")
    plt.title('Training Loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.plot(Train_loss)
    plt.savefig("Train_Vis.png")
    
    plt.title('Dev Accuracy')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy (%)')
    plt.plot(Test_acc)
    plt.savefig("Dev_Vis.png")
    
    # Writeout test labels
    labels = predictLabels(model, test_loader, device)
    np.save(hyper["testLabelName"], labels)
    labels = list(map(int, labels))
    idxs = np.array(list(range(len(labels))))
    labels = np.array(labels)
    df = pd.DataFrame({"id" : idxs, "label" : labels})
    df.to_csv(hyper["testLabelCSVfn"], index=False)

if __name__ == "__main__":
    hyper = {
        "nEpochs":5,
        "lr":0.0001,
        "lr_decayRate":0.0,
        "randomSeed":2,
        "kContext":12,
        "batchSize":256,
        "dataPath":'./data',
        "weightDirName": './checkpoint/',
        "testLabelName" : "./data/test_labels.npy",
        "testLabelCSVfn": "./data/test_labels.csv",
        "hiddenDims": [2048,1024,1024,1024,1024,512,256],
        "checkpointPath":"./checkpoint/optimContDecrease_Epoch_8contextK_v2.txt"
    }
    main(hyper)


# In[ ]:







