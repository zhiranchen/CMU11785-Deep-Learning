all:
	cp hw4/experiments/$(runid)/predictions-test-$(epoch).npy predictions.npy
	cp hw4/experiments/$(runid)/generated-$(epoch).txt generated.txt
	cp hw4/experiments/$(runid)/generated_logits-test-$(epoch).npy generated_logits.npy
	cp hw4/training.ipynb training.ipynb
	tar -cvf handin.tar training.ipynb predictions.npy generated.txt generated_logits.npy
	rm -f generated.txt predictions.npy training.ipynb generated_logits.npy
