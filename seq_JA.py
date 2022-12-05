#!/urs/bin/python
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class reditDataset(Dataset):
    def __init__(self, split, csv_dir):
        # read data
        csv = pd.read_csv(csv_dir,sep = '§')
        self.csv = csv[csv['split'] == split]


    def __len__(self):
        return len(self.csv)

    def __getitem__(self,idx):
        data = {}
        data['message'] = self.csv.iloc[idx]['Message']
        data['response'] = self.csv.iloc[idx]['Response']

        return data



# if __name__ == '__main__':
#     csv_dir = "data/reddit_casual_split.csv"
   
#     dataset = reditDataset('Train', csv_dir)
#     loader = DataLoader(dataset)
#     data = next(iter(dataset))
#     #print(type(data['message']))
#     #print(data['response'])
#     max_lenth =  0


#     for data in loader:

#         if len(data['message'][0].split(" ")) > 200:
#             max_lenth += 1 
#         #print(*data['message'])
#         #print(*data['response'])
#     print(max_lenth)

        
    

# import datasets
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
# from models import EncoderRNN, AttnDecoderRNN

device = 'cuda' if torch.cuda.is_available() else False

###
# Hyper parameters
##
BATCHSIZE = 16
EPOCHS = 100
LR = 1e-3

## Dataset
csv_dir = './data/reddit_casual_split.csv'
trainset = reditDataset('Train', csv_dir)
valset = reditDataset('Val', csv_dir)
trainset = reditDataset('Test', csv_dir)

trainloader = DataLoader(trainset, batch_size=BATCHSIZE)
valloader = DataLoader(trainset, batch_size=BATCHSIZE)
testloader = DataLoader(trainset, batch_size=BATCHSIZE)

#Encoder 
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
#Decoder
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
#attention
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

#data reader
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)
def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)
#slicer for mini batches
get_slice = lambda i, size: range(i * size, (i + 1) * size)

## Model
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
# encoder = EncoderRNN() # NOTE missing parameters!
# decoder = AttnDecoderRNN()

encoder_optimizer = optim.SGD(encoder.parameters(), lr=LR)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=LR)

criterion = nn.NLLLoss()


#One Traning loop
def train_iter(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

#Validation
def val_iter(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    return loss.item() / target_length

def train(EPOCHS,encoder,decoder):

    best_loss = 100

    for epoch in range(EPOCHS):

        val_loss = 0 
        #mini batches
        slce = get_slice(mini_batch, total_batch_size)
        training_pairs = X_TRAIN[slce]

        for i in range(len(trainloader)):
            #split mini battch into input and target
            training_pair = trainloader[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]


            #train and find loss
            train_loss = train_iter(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        

 
        for i in range(len(valloader)):
        #split mini battch into input and target
            val_pair = valloader[iter - 1]
            input_tensor = val_pair[0]
            target_tensor = val_pair[1]
            #see loss on validation set
            val_loss += val_iter(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH)


        #if traning impores on validations set save model.
        if val_loss < best_loss:
            best_encoder = encoder
            best_decoder = decoder
            best_loss = val_loss

    return best_encoder, best_decoder
            



def test(encoder,decoder):


    test_loss =  val_iter(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH)
    # show examples
    pass



if __name__ == '__main__':
    train()
    test()