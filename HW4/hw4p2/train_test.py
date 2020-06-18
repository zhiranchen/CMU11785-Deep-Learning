import time
import torch
from plot import plot_grad_flow
### Add Your Other Necessary Imports Here! ###

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_mask(lens):
    lens = torch.tensor(lens).to(DEVICE)
    max_len = torch.max(lens)

    mask = (torch.arange(0, max_len).repeat(lens.size(0), 1).to(DEVICE) < \
                lens.unsqueeze(1).expand(lens.size(0), max_len)).int()
    return mask

def calc_edit_dist(preds, targets):
    res = 0.0
    for pred, target in zip(preds, targets):
        dist = Levenshtein.distance(pred, target)
#         print("Lev dist pred {}".format(pred))
#         print("Lev dist target {}".format(target))
#         print("Lev dist {}".format(dist))
        res += dist
    return res
def train(model, train_loader, criterion, optimizer, epoch, displayBatchFreq=50):
    model.train()
    model.to(DEVICE)
    start = time.time()
    runningLoss = 0.0
    runningPerplex = 0.0
    trainingLoss = 0.0
    trainingPerplex = 0.0

    # 1) Iterate through your loader
    for batch_idx, (data, target, dataLens, targetLens) in enumerate(train_loader):
        # 2) Use torch.autograd.set_detect_anomaly(True) to get notices about gradient explosion
        with torch.autograd.set_detect_anomaly(False): # close to 
            # 3) Set the inputs to the device.
            data, target, dataLens, targetLens = data.to(DEVICE), target.to(DEVICE), dataLens.to(DEVICE), targetLens.to(DEVICE)
            optimizer.zero_grad()

            # 4) Pass your inputs, and length of speech into the model.
            predictions = model(speech_input=data, speech_len=dataLens, batchNum=batch_idx, text_input=target, isTrain=True)
            #print("prediction size {}".format(predictions.size()))

            # 5) Generate a mask based on the lengths of the text to create a masked loss.
            # 5.1) Ensure the mask is on the device and is the correct shape.
            mask = generate_mask(targetLens).to(DEVICE)
            #print("mask size {}".format(mask.size()))


            # 6) If necessary, reshape your predictions and origianl text input
            # 6.1) Use .contiguous() if you need to.

            # 7) Use the criterion to get the loss.
            #print("Loss input")
            #print("Predictions size {}".format(predictions.size()))
            #print("Target size {}".format(target.size()))
            loss = criterion(predictions.view(-1, predictions.size(2)), target.view(-1))

            # 8) Use the mask to calculate a masked loss.
            #print("Loss Size {}".format(loss.size()))
            #print("mask.view(-1) size {}".format(mask.view(-1).size()))
            masked_loss = torch.sum(loss * mask.view(-1)) / torch.sum(mask)
            #print("Masked_loss size {}".format(masked_loss.size()))
            #masked_loss = loss.sum() / targetLens.sum()
            ## Cumulate running loss and perplexity
            currLoss = masked_loss.item()
            currPerplex = torch.exp(masked_loss).item()
            runningLoss += currLoss
            runningPerplex += currPerplex
            trainingLoss += currLoss
            trainingPerplex += currPerplex

            # 9) Run the backward pass on the masked loss.
            masked_loss.backward()
            ## Plot gradient flow

            # 10) Use torch.nn.utils.clip_grad_norm(model.parameters(), 2)
            torch.nn.utils.clip_grad_norm(model.parameters(), 2)

            # 11) Take a step with your optimizer
            optimizer.step()

            # 12) Normalize the masked loss


            # 13) Optionally print the training loss after every N batches
            if batch_idx % displayBatchFreq == (displayBatchFreq-1):
                plt = plot_grad_flow(model.named_parameters())
                plt.savefig('./grad_plot/epoch{}_batch{}'.format(epoch, batch_idx), bbox_inches='tight')
            if batch_idx % displayBatchFreq == (displayBatchFreq-1):
                print("Epoch: {} Batch: {}\tLoss: {:.5f}\tCurrPerplex: {:.5f}\tAvgPreplex:{:.5f}\tTimestamp: {:.5f}".format(epoch,\
                      batch_idx, runningLoss/displayBatchFreq,\
                      currPerplex, runningPerplex/displayBatchFreq,
                      time.time() - start))
                runningLoss = 0.0
                runningPerplex = 0.0

            del data
            del target
            del dataLens
            del targetLens
            torch.cuda.empty_cache()

    end = time.time()
    print("Finished Epoch: {}\tTrainLoss: {:.5f}\tTrainPerplex: {:.5f}\tTimeTook: {:.5f}".format(epoch,\
          trainingLoss/len(train_loader), trainingPerplex/len(train_loader), end - start))

def val(model, test_loader, criterion, epoch, displayBatchFreq=50, displayPredFreq=10):
    ### Write your test code here! ###
    model.eval()
    model.to(DEVICE)
    start = time.time()
    runningLoss = 0.0
    runningPerplex = 0.0
    runningDist = 0.0
    testLoss = 0.0
    testPerplex = 0.0
    numSeq = 0.0
    print(len(test_loader))
    for batch_idx, (data, target, dataLens, targetLens) in enumerate(test_loader):
        data, target, dataLens, targetLens = data.to(DEVICE), target.to(DEVICE), dataLens.to(DEVICE), targetLens.to(DEVICE)
        
        predictions = model(speech_input=data, speech_len=dataLens, batchNum=batch_idx, text_input=None, isTrain=False)

        #mask = generate_mask(targetLens).to(DEVICE)#torch.arange(target.size(1)).unsqueeze(0).to(DEVICE) >= targetLens.unsqueeze(1)
        
        #loss = criterion(predictions.view(-1, predictions.size(2)), target.view(-1))
        
        #masked_loss = torch.sum(loss * mask.view(-1)) / torch.sum(mask)
        
        #runningLoss += masked_loss.item()
        #runningPerplex += torch.exp(masked_loss).item()
        #testLoss += masked_loss.item()
        #testPerplex += torch.exp(masked_loss).item()

#         if batch_idx % displayBatchFreq == (displayBatchFreq-1):
#             print("Epoch: {} Batch: {}\tLoss: {:.5f}\tPerplex: {:.5f}\tTimestamp: {:.5f}".format(epoch,\
#                   batch_idx, runningLoss/displayBatchFreq,\
#                   runningPerplex/displayBatchFreq,
#                   time.time() - start))
#             runningLoss = 0.0
#             runningPerplex = 0.0
        
        # Compare validation edit distance

        predText = transform_index_to_letter(predictions.argmax(-1).detach().cpu().numpy(),\
                                                 [letter2index['<eos>'], letter2index['<pad>']])
        targetText = transform_index_to_letter(target.detach().cpu().numpy(),\
                                               [letter2index['<eos>'], letter2index['<pad>']])
        
        runningDist += calc_edit_dist(predText, targetText)
        numSeq += len(predText)
        
        if batch_idx % displayPredFreq == (displayPredFreq-1):
            print("-"*20)
            print("Pred:\n{}\nTarget:\n{}\n".format(predText[0], targetText[0]))
            print("-"*20)
            
        del data
        del target
        del dataLens
        del targetLens
        torch.cuda.empty_cache()
    end = time.time()
    print("Finished Epoch: {}\tEditDist: {:.5f}\tTimeTook: {:.5f}".format(epoch, runningDist/numSeq, end - start))
    return runningDist/numSeq

def inference(model, data_loader, hyper, isValid=False):
    res = []
    with torch.no_grad():
        model.eval()
        model.to(DEVICE)
        start = time.time()
        if isValid:
            targetRes = []
            runningDist = 0.0
            numSeq = 0.0
            for batch_idx, (data, target, dataLens, targetLens) in enumerate(data_loader):
                data, target, dataLens, targetLens = data.to(DEVICE), target.to(DEVICE), dataLens.to(DEVICE), targetLens.to(DEVICE)
                predictions = model(speech_input=data, speech_len=dataLens, batchNum=batch_idx, text_input=None, isTrain=False)
                
                # Compare validation edit distance

                predText = transform_index_to_letter(predictions.argmax(-1).detach().cpu().numpy(),\
                                                         [letter2index['<eos>'], letter2index['<pad>']])
                targetText = transform_index_to_letter(target.detach().cpu().numpy(),\
                                                       [letter2index['<eos>'], letter2index['<pad>']])
                res.extend(predText)
                targetRes.extend(targetText)

                runningDist += calc_edit_dist(predText, targetText)
                numSeq += len(predText)

                if batch_idx % 5 == (5-1):
                    print("-"*20)
                    print("Pred:\n{}\nTarget:\n{}\n".format(predText[0], targetText[0]))
                    print("-"*20)
            print("Edit distance for VAL:\t{:.5f}".format(runningDist/numSeq))
            df = pd.DataFrame({"Id" : np.array(list(range(len(res)))), "Predicted" : np.array(res), "Target": np.array(targetRes)})
            df.to_csv(hyper['devPredCSVfn'], index=False)
            return df
        else:
            for batch_idx, (data, dataLens) in enumerate(data_loader):
                data, dataLens = data.to(DEVICE), dataLens.to(DEVICE)
                predictions = model(data, dataLens, batch_idx, text_input=None, isTrain=False)
                predTexts = transform_index_to_letter(predictions.argmax(-1).detach().cpu().numpy(),\
                                                         [letter2index['<eos>'], letter2index['<pad>']])
                res.extend(predTexts)
    
            idxs = np.array(list(range(len(res))))
            preds = np.array(res)
            np.save(hyper['testPredNpyfn'], preds)
            df = pd.DataFrame({"Id" : idxs, "Predicted" : preds})
            df.to_csv(hyper['testPredCSVfn'], index=False)
            return df
