File structure:
- HW1P2
	- main.py
	- data
		- train.npy
		- train_labels.npy
		- dev.npy
		- dev_labels.npy
		- test.npy
		- test_labels_v2.npy
		- test_labels_v2.csv
	- checkpoint
		- optim_Epoch_4contextK_v2.txt
		- optimCont_Epoch_8contextK_v2.txt
		- optimContDecrease_Epoch_8contextK_v2.txt

To run my model:
	- Have the file structure as above
	- Type "source activate pytorch_p36" in the terminal
	- In the HW1P2 directory, type "python3 main.py" to run the model
	- After the model finishes, find predicted test_labels_v2.npy and csv files under data folder
Dataloader design:
	- Used a python list to store (i, j) to index into into the list, where i is the utterance
	  index and j is the frame index. I used python.take to include 12 left and right frames of
	  the current indexed frame.

Steps taken to train my model that gets the optimal result:
- First train model with configuration of the following structure
	hidden layer dimensions: [2048,1024,1024,1024,1024,512,256]
	MLP(
	  (net): Sequential(
	    (0): Linear(in_features=1000, out_features=2048, bias=True)
	    (1): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	    (2): LeakyReLU(negative_slope=0.01)
	    (3): Linear(in_features=2048, out_features=1024, bias=True)
	    (4): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	    (5): LeakyReLU(negative_slope=0.01)
	    (6): Linear(in_features=1024, out_features=1024, bias=True)
	    (7): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	    (8): LeakyReLU(negative_slope=0.01)
	    (9): Linear(in_features=1024, out_features=1024, bias=True)
	    (10): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	    (11): LeakyReLU(negative_slope=0.01)
	    (12): Linear(in_features=1024, out_features=1024, bias=True)
	    (13): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
	    (14): LeakyReLU(negative_slope=0.01)
	    (15): Linear(in_features=1024, out_features=512, bias=True)
	    (16): LeakyReLU(negative_slope=0.01)
	    (17): Linear(in_features=512, out_features=256, bias=True)
	    (18): LeakyReLU(negative_slope=0.01)
	    (19): Linear(in_features=256, out_features=138, bias=True)
	  )
	)
	kContext size: 12
	optimizer: Adam with 0.001 learning rate with learning rate scheduler
	learning rate scheduler: ReduceLROnPlateau, mode=min, patientce=2
	batch size: 256
	number of Epochs: 5
	Save the model in file optim_Epoch_4contextK_v2.txt
- Then load the model from optim_Epoch_4contextK_v2.txt and continue training same expect
	number of Epochs: 9
	Save the model in file optimCont_Epoch_8contextK_v2.txt
- Then load the model fin file optimCont_Epoch_8contextK_v2.txt and continue trainign with
	number of Epochs: 9
	Save the model in file optimContDecrease_Epoch_8contextK_v2.txt

Reference Links:
	https://florimond.dev/blog/articles/2018/10/reconciling-dataclasses-and-properties-in-python/
	https://medium.com/@Biboswan98/optim-adam-vs-optim-sgd-lets-dive-in-8dbf1890fbdc
	https://pytorch.org/docs/stable/optim.html
	https://stackoverflow.com/questions/41153803/zero-padding-slice-past-end-of-array-in-numpy
	https://dev.to/hardiksondagar/how-to-use-aws-ebs-volume-as-a-swap-memory-5d15
	https://www.google.com/search?client=safari&sxsrf=ACYBGNSjEyEM_L_y5jQjSiKA7LAUMCu38g%3A1581151963058&source=hp&ei=23Y-Xosv5q_K0w_e2IjoAg&q=torch.save%28model.state_dict%28%29&oq=torch.save%28model.state_dict%28%29&gs_l=psy-ab.3..0l3j0i333l3.4254.4254..4482...3.0..0.85.85.1......0....2j1..gws-wiz.yz9xThPrXqE&ved=0ahUKEwjLiN2IysHnAhXml3IEHV4sAi0Q4dUDCAs&uact=5
	https://pytorch.org/tutorials/beginner/saving_loading_models.html
	https://stackoverflow.com/questions/33492260/save-multiple-arrays-to-a-csv-file-with-column-names
	https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
