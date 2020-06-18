File structure:
	- hw3p2 - p2.py
	- hw3p2
	- provided file and folders including data from gaggle
	- checkpoint
	- CNN1_Cont_Epoch6.txt
	- data
	- Saved prediction of test data

To run my model:
	- Have the file structure as above
	- Type “source activate pytorch_p36” in the terminal
	- In the hw3p2 top directory, type “python3 p2.py” to run the model
	- After the model finishes, find predicted predicted.np and predicted.csv files under the data folder

Design choices:
	- Used one conv1d CNN with (input, hiddenSize) = (40, 256) with kernel size=3, padding=1, stride=1, bias=False. Followed by a BatchNorm1d and ReLU. Then followed by 3 stacked BiLSTM layers each of 256 units with dropout rate = 0.2 to avoid overfitting to the training data. Then followed by one linear layer with dim (2*256, 256) and another linear layer with dim (256, 37)
	- Used Adam optimizer with lr=1e-3 and weight decay=5e-5 (following the baseline advise)
	- Used a ReduceLROnPlateau scheduler with factor = 0.5, patience=1
	- Used CTCLoss
	- Trained with the above configuration for 30 epochs.