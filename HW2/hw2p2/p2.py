# Import all the necessary libraries
import PIL
import numpy as np
import torch
import sys
import os
import time
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score


class MyDatasetTestClassify(Dataset):
	"""
	Dataset instance of classification task, test data
	"""
    def __init__(self, testFN, testImgFolderPath):
        self.imgFolderPath = testImgFolderPath
        with open(testFN) as f:
            self.fileList = [line.rstrip() for line in f]
    def __len__(self):
        return len(self.fileList)
    def __getitem__(self, idx):
        img = PIL.Image.open(self.imgFolderPath + self.fileList[idx])
        img = transforms.ToTensor()(img)
        return img, -1
    def getFileList(self):
        return self.fileList


class MyDatasetVerify(Dataset):
	"""
	Dataset instance of verification task
	"""
    def __init__(self, pairFN, imgFolderPath):
        self.imgFolderPath = imgFolderPath
        with open(pairFN) as f:
            self.pairList = [line.rstrip() for line in f]
    def __len__(self):
        return len(self.pairList)
    def __getitem__(self, idx):
        items = self.pairList[idx].split()
        fn1, fn2 = items[0], items[1]
        img1 = PIL.Image.open(self.imgFolderPath + fn1)
        img2 = PIL.Image.open(self.imgFolderPath + fn2)
        img1 = transforms.ToTensor()(img1)
        img2 = transforms.ToTensor()(img2)
        if len(items) == 3: # validation
            return img1, img2, int(items[2])
        else: # test
            return img1, img2, -1
    def getPairList(self):
        return self.pairList


class BottleNeck(nn.Module):
	"""
	Bottleneck block fo MobileNetV2
	"""
    def __init__(self, inChannel, outChannel, stride, expandT):
        super(BottleNeck, self).__init__()
        self.stride = stride
        self.shouldSkip = self.stride == 1 and inChannel == outChannel
        hiddenDim = int(inChannel * expandT)
        self.conv = nn.Sequential(
            # 1 x 1 expansion layer + bn + ReLU6
            nn.Conv2d(inChannel, hiddenDim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hiddenDim),
            nn.ReLU6(inplace=True),
            # 3 x 3 depthwise conv + bn + ReLU6
            nn.Conv2d(hiddenDim, hiddenDim, 3, self.stride, 1, groups=hiddenDim, bias=False),
            nn.BatchNorm2d(hiddenDim),
            nn.ReLU6(inplace=True),
            # 1 x 1 projection layer + bn
            nn.Conv2d(hiddenDim, outChannel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outChannel)
        )
    
    def forward(self, x):
        # only skip connnection when stride==1 and inChannel==outChannel
        if self.shouldSkip:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Network(nn.Module):
	"""
	MobileNetV2 implementation
	"""
    def __init__(self, inputSize, bottlesSetting, numClasses, feat_dim=1280):
        super(Network, self).__init__()
        block = BottleNeck
        firstChannel = 32
        lastChannel = 1280
        blocks = [conv2d_3x3_bn_relu(3, firstChannel, 1)]
        # build MobileNet bottlenecks
        bottleInChannel = firstChannel
        for t, c, n, s in bottlesSetting:
            bottleOutChannel = c
            for i in range(n):
                if i == 0: # the first layer in a sequence
                    blocks.append(block(bottleInChannel, bottleOutChannel, s, t))
                else:
                    blocks.append(block(bottleInChannel, bottleOutChannel, 1, t))
                bottleInChannel = bottleOutChannel
        # build the conv2d 1x1 layer
        blocks.append(conv2d_1x1_bn_relu(bottleInChannel, lastChannel))
        self.net = nn.Sequential(*blocks)
        
        # built classifier
        self.classifier = nn.Linear(lastChannel, numClasses)
         
    def forward(self, x):
        x = self.net(x)
        output = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1) # flatten
        classification_out = self.classifier(output)
        embedding_out = output
        return embedding_out, classification_out


def conv2d_3x3_bn_relu(inChannel, outChannel, stride):
	"""
	Conv2d layer with 3x3 kenel and customized stride + batchnorm + relu
	"""
    return nn.Sequential(
        nn.Conv2d(inChannel, outChannel, 3, stride, 1, bias=False),
        nn.BatchNorm2d(outChannel),
        nn.ReLU6(inplace=True)
    )


def conv2d_1x1_bn_relu(inChannel, outChannel):
	"""
	Conv2d layer with 1x1 kenel and 1 stride + batchnorm + relu
	"""
    return nn.Sequential(
        nn.Conv2d(inChannel, outChannel, 1, 1, 0, bias=False),
        nn.BatchNorm2d(outChannel),
        nn.ReLU6(inplace=True)
    )


def getLoaders(trainDS, devDS, testDS, batchS):
	"""
	Create and return dataloader for train, dev, test dataset
	"""
    print("*** Create data loader ***")
    
    # Train
    loader_args = dict(shuffle=True, batch_size=batchS, num_workers=8, pin_memory=True)
    train_loader = DataLoader(trainDS, **loader_args)
    
    # Dev
    dev_loader = DataLoader(devDS, **loader_args)
    
    # Test
    test_loader_args = dict(shuffle=False, batch_size=100, num_workers=1, pin_memory=True)
    test_loader = DataLoader(testDS, **test_loader_args)
    
    return train_loader, dev_loader, test_loader


def train_epoch(model, data_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        data, target = data.cuda(), target.cuda()

        outputs = model(data)[1]
        loss = criterion(outputs, target)
        running_loss += loss.item()
        
        loss.backward()
        optimizer.step()

        if batch_idx % 1000 == 0:
            print("Epoch: {}\tBatch: {}\tTimestamp: {}".format(epoch, batch_idx, time.time() - start_time))
        
        # clear computation cache
        torch.cuda.empty_cache()
        del data
        del target
        del loss
    end_time = time.time()
    running_loss = running_loss / len(data_loader)
    return running_loss


def testClassify(model, test_loader, epoch):
    with torch.no_grad():
        model.eval()
        start_time = time.time()
        running_loss = 0.0
        correct_predictions = 0.0
        total_predictions = 0.0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()

            outputs = model(data.float())[1]
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, target.long()).detach()
            total_predictions += target.size(0)
            correct_predictions += (predicted==target).sum().item()
            running_loss += loss.item()
            if batch_idx % 500 == 0:
                print("Epoch: {}\tBatch: {}\tTimestamp: {}".format(epoch, batch_idx, time.time()-start_time))
            del data
            del target
        running_loss /= len(test_loader)
        acc = (correct_predictions/total_predictions)*100.0
        return running_loss, acc


def testVerify(model, vLoader):
    similarity = np.array([])
    true = np.array([])
    start_time = time.time()
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, (imgs1, imgs2, targets) in enumerate(vLoader):
            imgs1, imgs2, targets = imgs1.cuda(), imgs2.cuda(), targets.cuda()
            # find cos similarity between embeddings
            imgs1Embed = model(imgs1.float())[0]
            imgs2Embed = model(imgs2.float())[0]
            sim = F.cosine_similarity(imgs1Embed, imgs2Embed) 
            similarity = np.concatenate((similarity, sim.cpu().numpy().reshape(-1)))
            true = np.concatenate((true, targets.cpu().numpy().reshape(-1)))
            if batch_idx % 100 == 0:
                print("Batch: {}\t Timestamp:{}".format(batch_idx, time.time()-start_time))
            del imgs1
            del imgs2
            del targets
    return similarity, true
            

def predictLabels(model, test_loader):
    with torch.no_grad():
        model.eval()
        res = np.array([])
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()

            outputs = model(data)[1]
            _, predicted = torch.max(outputs.data, dim=1)
            res = np.concatenate((res, predicted.cpu().numpy().reshape(-1)))
            del data
            del target
    return res
        


def main(hyper):
	dataFolder = hyper['dataPath']
	wegithDirName = hyper['weightDirName']
	cuda = torch.cuda.is_available()

	print("*** Load raw data ***")
	train = datasets.ImageFolder(root=dataFolder+"/train_data/medium", transform=transforms.Compose([
	                                transforms.RandomHorizontalFlip(), transforms.ToTensor()]))
	dev = datasets.ImageFolder(root=dataFolder+"/validation_classification/medium",
	                           transform=transforms.ToTensor())
	# Load custom dataset for test since it does not follow ImageFolder structure
	test = MyDatasetTestClassify(hyper["classifyTestListPath"], hyper["classifyTestImgFolderPath"])
	print("train data stat: {} images \t {} classes".format(train.__len__(), len(train.classes)))
	print("dev data stat: {} images \t {} classes".format(dev.__len__(), len(dev.classes)))
	print("test data stat: {} images".format(test.__len__()))

	# Get data loaders
	train_loader, dev_loader, test_loader = getLoaders(train, dev, test, hyper["batchSize"])

	# Create the model and define the Loss and Optimizer
	print("*** Create the model and define  Loss and Optimizer ***")
	inputSize = train.__len__()         # number of train input images
	outputSize = len(train.classes)     # number of unique face classes
	model = Network(inputSize, hyper["bottleneckSetting"], outputSize)
	checkpoint = torch.load(hyper["checkpoingPath"])
	model.load_state_dict(checkpoint["model_state_dict"])
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=hyper["lr"], momentum=0.9, nesterov=True, weight_decay=5e-4)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=1)
	device = torch.device("cuda" if cuda else "cpu")
	model.cuda()
	print(model)

	# Train the model for N epochs
	print("*** Train the model for N epochs ***")
	Train_loss = []
	Train_acc = []
	Test_loss = []
	Test_acc = []
	for i in range(hyper["nEpochs"]):
	    for prarm_group in optimizer.param_groups:
	        print("Current lr: \t{}".format(prarm_group["lr"]))
	    startTime = time.time()
	    print("Train\tEpoch: {}".format(i))
	    train_loss = train_epoch(model, train_loader, criterion, optimizer, i)
	    if hyper["task"] == "Classification":
	        print("Classify Train \tEpoch: {}".format(i))
	        train_loss, train_acc = testClassify(model, train_loader, i)
	        print("Classify Dev \tEpoch: {}".format(i))
	        dev_loss, dev_acc = testClassify(model, dev_loader, i)
	        print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
	              format(train_loss, train_acc, dev_loss, dev_acc))
	    else:
	        print("Verification task")
	    scheduler.step(dev_loss)
	    Train_loss.append(train_loss)
	    Train_acc.append(train_acc)
	    Test_loss.append(dev_loss)
	    Test_acc.append(dev_acc)
	    print("*** Saving Checkpoint ***")
	    path = "{}ContContContInitWeight_BaselineSGD_StepLR_Epoch{}.txt".format(wegithDirName, i)
	    torch.save({
	        "epoch": i,
	        'model_state_dict': model.state_dict(),
	        'optimizer_state_dict': optimizer.state_dict(),
	        'train_loss': train_loss,
	        "train_acc": train_acc,
	        'dev_loss':dev_loss,
	        'dev_acc': dev_acc
	    }, path)
	    print("="*20 + " Epoch {} took {}s".format(i, time.time()-startTime) + "="*20)


	# Writeout test labels for classification task
	labels = predictLabels(model, test_loader)
	labels = list(map(int, labels))
	idxs = np.array(test.getFileList())
	labels = np.array(labels)
	# create mappings of file set labels to true labels
	alphabetSorted = sorted([str(x) for x in range(0, 2300)])
	filesetTrueLabelTuple = [(i, int(alphabetSorted[i])) for i in range(len(alphabetSorted))]
	mapping = dict(filesetTrueLabelTuple)
	labels = np.array(list(map(mapping.get, labels)))
	np.save(hyper["testClassLabelName"], labels)
	df = pd.DataFrame({"Id" : idxs, "Category" : labels})
	df.to_csv(hyper["testClassLabelCSVfn"], index=False)

	# Read in verification pairs for validation
	verifyData_valid = MyDatasetVerify(hyper["verifyPairListPath"], hyper["verifyImgFolderPath"])
	verify_loader_args_valid = dict(shuffle=False, batch_size=200, num_workers=8, pin_memory=True)
	verify_loader_valid = DataLoader(verifyData_valid, **verify_loader_args_valid)
	# Calculate simliarity score
	cosScore_valid, trueScore_valid = testVerify(model, verify_loader_valid)
	# Report AUC
	auc = roc_auc_score(trueScore_valid, cosScore_valid)
	print("*** AUC: {} ***".format(auc))

	# Read in verification pairs for test
	verifyData_test = MyDatasetVerify(hyper["verifyTestPairListPath"], hyper["verifyTestImgFolderPath"])
	verify_loader_args_test = dict(shuffle=False, batch_size=300, num_workers=8, pin_memory=True)
	verify_loader_test = DataLoader(verifyData_test, **verify_loader_args_test)
	# Calculate similarity score
	cosScore_test, _ = testVerify(model, verify_loader_test)

	# Save predictied similarity
	cosScore_test = np.array(cosScore_test)
	np.save(hyper["testVeriLabelName"], cosScore_test)
	trial = np.array(verifyData_test.getPairList())
	df = pd.DataFrame({"trial" : trial, "score" : cosScore_test})
	df.to_csv(hyper["testVeriLabelCSVfn"], index=False)

if __name__ == "__main__":
	# Hyperparameters
	hyper = {
	    "task": "Classification",
	    "bottleneckSetting": [[1, 16, 1, 1], # t, c, n, s
	                          [6, 24, 2, 1],
	                          [6, 32, 3, 2],
	                          [6, 64, 4, 1],
	                          [6, 96, 3, 2],
	                          [6, 160, 3, 1],
	                          [6, 320, 1, 1]],
	    "nEpochs":50,
	    "batchSize":256,
	    "lr":0.001,#1e-3,
	    "dataPath": "./11-785hw2p2-s20",
	    "checkpoingPath": "./checkpoint/ContContInitWeight_BaselineSGD_StepLR_Epoch4.txt",
	    "classifyTestImgFolderPath": "./11-785hw2p2-s20/test_classification/medium/",
	    "classifyTestListPath": "./11-785hw2p2-s20/test_order_classification.txt",
	    "verifyImgFolderPath": "./11-785hw2p2-s20/validation_verification/",
	    "verifyPairListPath": "./11-785hw2p2-s20/validation_trials_verification.txt",
	    "verifyTestPairListPath": "./11-785hw2p2-s20/test_trials_verification_student.txt",
	    "verifyTestImgFolderPath": "./11-785hw2p2-s20/test_verification/",
	    "weightDirName": "./checkpoint/",
	    "testClassLabelName":"./output/test_class_labels.npy",
	    "testClassLabelCSVfn":"./output/test_class_labels.csv",
	    "testVeriLabelName":"./output/test_veri_labels.npy",
	    "testVeriLabelCSVfn":"./output/test_veri_labels.csv",
	}
	main(hyper)



