File structure:
	- main.py
	- models.py
	- plot.py
	- train_test.py
	- dataloader.py
	- attention (folder)
	- checkpoint (folder) - data (folder)
	- train_new.npy
	- dev_new.npy
	- test_new.npy
	- train_transcripts.npy - dev_transcripts.npy - predicted_test.csv
	- predicted_dev.csv


To run my model:
	- Have the file structure as above 
	- Type “source activate pytorch_p36” in the terminal 
	- In the top directory, type “python3 main.py” to run the model 
	- After the model finishes, find predicted predicted_test.csv, predicted_dev.csv in the data folder


Design choices:
	- Experimented with gumbel noise, changing teacher forcing rate from 0.1 and gradually to 0.4, use ReduceOnLRPlateau schedule
	- Used Adam optimizer with lr=0.001
	- Used a ReduceLROnPlateau scheduler with factor = 0.75, patience=1, threshold=0.01
	- First train the model start with teacher forcing rate at 0.1, after 25 epochs, when the edit
	distance stop improving, change teacher forcing rate to 0.2, then train for another 30
	epochs
	- Gumbel noise does not help my performance somehow? the model stopped improving at a
	very early stage.
	- Changing teacher forcing rate worked well for me 