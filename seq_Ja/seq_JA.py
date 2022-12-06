# !/urs/bin/python
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from dataset import redditDataset
import torch
from io import open
import unicodedata
import string
import re
import random
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models import EncoderRNN, AttnDecoderRNN
from vocabulary import Voc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###
# Hyper parameters
##
BATCHSIZE = 16
EPOCHS = 10
LR = 1e-3
max_length = 3
hidden_size = 256
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

## Dataset
csv_dir = './data/reddit_casual_split.csv'
trainset = redditDataset('Train',max_length, csv_dir)
valset = redditDataset('Val', max_length,csv_dir)
testset = redditDataset('Test',max_length ,csv_dir)

trainloader = DataLoader(trainset, batch_size=BATCHSIZE,shuffle=True)
valloader = DataLoader(valset, batch_size=BATCHSIZE)
testloader = DataLoader(testset, batch_size=BATCHSIZE)


    

model_path = "./seq_Ja/modelsaves/best_"

voc = Voc("reddit")

def voc_build():
    for message, response in trainloader:

        for i in range(len(message)):
            voc.addSentence(message[i])
            voc.addSentence(response[i])
    for message, response in valloader:

        for i in range(len(message)):
            voc.addSentence(message[i])
            voc.addSentence(response[i])
    for message, response in testloader:

        for i in range(len(message)):
            voc.addSentence(message[i])
            voc.addSentence(response[i])
    print("Counted words:", voc.num_words)
    return voc
voc_build()


#data embeding
def sentence_to_index(sentence):
    indexes = [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

## Model
encoder = EncoderRNN(int(voc.num_words),int(hidden_size)).to(device)
decoder = AttnDecoderRNN(hidden_size, voc.num_words, dropout_p=0.1,max_length=max_length).to(device)

encoder_optimizer = optim.SGD(encoder.parameters(), lr=LR)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=LR)

criterion = nn.NLLLoss()

#One Traning loop
def train_iter():
    encoder.train()
    decoder.train()

    teacher_forcing_ratio = 0.5
    loss_total = 0
    n=0
    for message, response in trainloader:

        loss = 0
        for i in range(len(message)):
            message_one = sentence_to_index(message[i])
            response_one = sentence_to_index(response[i])

            encoder_hidden = encoder.initHidden()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            input_length = message_one.size(0)
            
            target_length = response_one.size(0)

            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

            

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(
                    message_one[ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=device)

            decoder_hidden = encoder_hidden

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    loss += criterion(decoder_output, response_one[di])
                    decoder_input = response_one[di]  # Teacher forcing

            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input

                    loss += criterion(decoder_output, response_one[di])
                    if decoder_input.item() == EOS_token:
                        break

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    return loss.item() / len(trainloader)

#Validation
def val_iter(set):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for message, response in set:

            loss = 0
            for i in range(len(message)):

                message_one = sentence_to_index(message[i])
                response_one = sentence_to_index(response[i])

                encoder_hidden = encoder.initHidden()

                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                input_length = message_one.size(0)

                target_length = response_one.size(0)

                encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

                for ei in range(input_length):
                    encoder_output, encoder_hidden = encoder(
                        message_one[ei], encoder_hidden)
                    encoder_outputs[ei] = encoder_output[0, 0]

                decoder_input = torch.tensor([[SOS_token]], device=device)

                decoder_hidden = encoder_hidden

                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze().detach()  # detach from history as input

                    loss += criterion(decoder_output, response_one[di])
                    if decoder_input.item() == EOS_token:
                        break

        return loss.item() / len(set)


def train():

    best_loss = 777

    train_loss = 0
    val_loss = 0 

    for epoch in range(EPOCHS):

        
 
        #train and find loss
        train_loss = train_iter()
        
        val_loss = val_iter(valloader)


        #if traning impores on validations set save model.
        if val_loss < best_loss:

            torch.save(encoder.state_dict(), model_path + "encoder.pth")
            torch.save(decoder.state_dict(),model_path + "decoder.pth")

            best_loss = val_loss
            best_epoc = epoch

        print("current epoc :{}, current train loss :{} , current val loss :{}\n".format(epoch+1,train_loss, val_loss,))

        print("best epoc :{}, best val loss:{}\n".format(best_epoc+1,best_loss))
 
            



def test():
    encoder = torch.load(model_path + "encoder.pth")
    decoder = torch.load(model_path + "decoder.pth")


    test_loss =  val_iter(testloader)
    print(test_loss)
    # show examples
    pass



if __name__ == '__main__':
    train()
    test()