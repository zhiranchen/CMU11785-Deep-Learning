## Table of Contents
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [HW1 MLP | Phoneme Recognition](#hw1-mlp--phoneme-recognition)
- [HW2 CNN | Face Recognition and Verification](#hw2-cnn--face-recognition-and-verification)
- [HW3 RNN - Forward/Backword/CTC Beamsearch | Connectionist Temporal Classification](#hw3-rnn---forwardbackwordctc-beamsearch--connectionist-temporal-classification)
- [HW4 Word-level Neural Language Models using RNNs | Attention Mechanisms and Memory Networks](#hw4-word-leve-neural-language-models-using-rnns--attention-mechanisms-and-memory-networks)

## Introduction
This repo contains course project of [11785 Deep Learning](http://deeplearning.cs.cmu.edu) at CMU. The projects starts off with MLPs and progresses into more complicated concepts like attention and seq2seq models. Each homework assignment consists of two parts. 
Part 1 is the Autolab software engineering component that involves engineering my own version of pytorch libraries, implementing important algorithms, and developing optimization methods from scratch. 
Part 2 is the Kaggle data science component which work on project on hot AI topics, like speech recognition, face recognition, and neural machine translation.


## HW1 MLP | Phoneme Recognition

- <b>HW1P1</b>
Implement simple MLP activations, loss, batch normalization. 

- <b>HW1P2</b>
Kaggle challenge: [Frame level classification of speech](https://www.kaggle.com/c/11-785-s20-hw1p2). <br>Using knowledge of feedforward neural networks and apply it to speech recognition task. The provided dataset consists of audio recordings (utterances) and their phoneme state (subphoneme) lables. The data comes from articles published in the Wall Street Journal (WSJ) that are read aloud and labelled using the original text.
The job is to identify the phoneme state label for each frame in the test dataset. It is important to note that utterances are of variable length.

## HW2 CNN | Face Recognition and Verification
- <b>HW2P1</b>
Implement NumPy-based Convolutional Neural Networks libraries.

- <b>HW2P2</b>
Kaggle challebge: [Face Classification](https://www.kaggle.com/c/11-785-s20-hw2p2-classification) & [Verification](https://www.kaggle.com/c/11-785-s20-hw2p2-verification) using Convolutional Neural Networks.<br>Given an image of a person’s face, the task of classifying the ID of the face is known as face classification. The input to the system will be a face image and the system will have to predict the ID of the face. The ground truth will be present in the training data and the network will be doing an
N-way classification to get the prediction. The system is provided with a validation set for fine-tuning the model.
## HW3 RNN - Forward/Backword/CTC Beamsearch | Connectionist Temporal Classification
- <b>HW3P1</b>
Implement RNNs and GRUs deep learning library like PyTorch.

- <b>HW3P2</b>
Kaggle challenge: [Utterance to Phoneme Mapping](https://www.kaggle.com/c/11-785-s20-hw3p2).<br>This challenge works with speech data. The contest uses unaligned labels, which means the correlation between the features and labels is not given explicitly and the model will have to figure this out by itself. Hence the data will have a list of phonemes for each utterance, but not which frames correspond to which phonemes.
The main task for this assignment will be to predict the phonemes contained in utterances in the test set. The training data does not contain aligned phonemes, and it is not a requirement to produce alignment for the test data.

## HW4 Word-level Neural Language Models using RNNs | Attention Mechanisms and Memory Networks
- <b>HW4P1</b>
Train a Recurrent Neural Network on the WikiText-2 Language Moldeling Dataset. This task uses reucurrent network to model and generate text, and uses various techniques to regularize recurrent networks and improve their performance.

- <b>HW4P2</b>
Kaggle challenge: [Deep Learning Transcript Generation with Attention](https://www.kaggle.com/c/11-785-s20-hw4p2). <br> In this challenge, use a combination of Recurrent Neural Networks (RNNs) / Convolutional Neural Networks (CNNs) and Dense Networks to design a system for speech to text transcription. End-to-end, the system should be able to transcribe a given speech utterance to its corresponding transcript.
