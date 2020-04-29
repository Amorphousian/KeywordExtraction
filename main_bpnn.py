# -*- coding: utf-8 -*-

from func_bpnn import KeyWord
import torch


# Create new instance
k = KeyWord(pathX='allJobDescription.html',
            pathY='job_skills_training.txt')

# Read in data
k.initialize()
# 8-2 split train-validation
k.datasplit(s=0.2)
# Tokenize
k.tokenize()
# Train word2vector
k.w2v(w2vsize=128)
# Find unique words
k.unique()
# Turn words into vectors
k.tensorize()
# Train neuron network to pick key words
k.model_train(mbs=128, hls=256, epo=200, opt=torch.optim.Adadelta)
# Plot loss and accuracy
k.graphplot()
# Fit data into neuron network
k.model_fit()
# Calculate F1 scores
k.f1score()
# For each job, match selected key words
# k.keymatch()
