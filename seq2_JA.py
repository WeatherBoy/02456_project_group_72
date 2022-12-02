#!/urs/bin/python

import datasets
import torch
from io import open
import unicodedata
import string
import re
import random
import datasets
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models import EncoderRNN, AttnDecoderRNN

device = 'cuda' if torch.cuda.is_available() else False

###
# Hyper parameters
##
BATCHSIZE = 16
EPOCHS = 100
LR = 1e-3

## Dataset
csv_dir = './data/'
trainset = datasets.reditDataset('Train', csv_dir)
valset = datasets.reditDataset('Val', csv_dir)
trainset = datasets.reditDataset('Test', csv_dir)

trainloader = DataLoader(trainset, batch_size=BATCHSIZE)
valloader = DataLoader(trainset, batch_size=BATCHSIZE)
testloader = DataLoader(trainset, batch_size=BATCHSIZE)


## Model
encoder = EncoderRNN() # NOTE missing parameters!
decoder = AttnDecoderRNN()

encoder_optimizer = optim.SGD(encoder.parameters(), lr=LR)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=LR)

def train_iter():
    pass

def val_iter(loader):
    pass

def train():
    best_loss = 100
    for epoch in range(EPOCHS):
        train_loss =train_iter()
        val_loss = val_iter(valloader)
        
        if val_loss < best_loss:
            best_loss = val_loss
            # save model



def test():
    test_loss = val_iter(testloader)
    # show examples
    pass



if __name__ == '__main__':
    train()
    test()


