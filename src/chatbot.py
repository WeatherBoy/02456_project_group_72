# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Python script for training the chatbot model
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Created by Group 72 on Sun Dec 04 2022

# NOTE: Before running
### This dataset should be downloaded: https://zissou.infosci.cornell.edu/convokit/datasets/movie-corpus/movie-corpus.zip
### This script assumes (per default) that you unzipped the above folder in a neighbouring directory
### called: "data".

#%% IMPORTS AND BASIC CONFIGURATIONS ##############################################################
import torch
from torch import nn
from torch import optim

import os

print(f"\n\nCurrent working directory1: \n{os.getcwd()}\n\n")

from data_processing.cornelMovie_preProcessing import loadPrepareData, trimRareWords, writeDataToPath
from models.seq2seq_model2 import EncoderRNN, LuongAttnDecoderRNN
from trainers.seq2seq_model_train import trainIters
from validaters.seq2seq_model_validate import GreedySearchDecoder, evaluateInput

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# Wanna see how the model performs?
EVALUATE = False


#%% HYPER PARAMETERS ##############################################################################
### Configure model
MODEL_NAME = 'cb_model'
ATTN_MODEL = 'dot'
# ATTN_MODEL = 'general'
# ATTN_MODEL = 'concat'
HIDDEN_SIZE = 500
ENCODER_N_LAYERS = 2
DECODER_N_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 16

### Configure training/optimization
EPOCHS = 100
CLIPPING = 50.0
TEACHER_FORCING_RATIO = 1.0
LR = 0.0001
DECODER_LEARNING_RATIO = 5.0
N_ITERATIONS = 4000
PRINT_EVERY = 1
SAVE_EVERY = 500

### Configure data parameters
MAX_LENGTH = 10  # Maximum sentence length to consider
MIN_COUNT = 3    # Minimum word count threshold for trimming

### Change to path that matches where you put the data.
### I had the data in a neighbouring directory
DATA_NAME = "movie-corpus"
DATA_PATH = "./data/" + DATA_NAME
#DATA_PATH = "../data/" + DATA_NAME


#%% MAIN ##########################################################################################
if __name__ == '__main__':
    ### PROCESSING DATA ###########################################################################
    datafile = writeDataToPath(data_path=DATA_PATH)

    # Load/Assemble voc and pairs
    SAVE_DIR = "../modelWeights"
    voc, pairs = loadPrepareData(corpus=DATA_PATH,
                                corpus_name=DATA_NAME,
                                datafile=datafile,
                                save_dir=SAVE_DIR,
                                max_length=MAX_LENGTH
                                )

    # Trim voc and pairs
    pairs = trimRareWords(voc, pairs, MIN_COUNT)
    
    ### LOADING MODELS ###########################################################################
    loadFilename = os.path.join(SAVE_DIR, MODEL_NAME, DATA_NAME,
                           '{}-{}_{}'.format(ENCODER_N_LAYERS, DECODER_N_LAYERS, HIDDEN_SIZE),
                            '{}_checkpoint.tar'.format(N_ITERATIONS))
    
    loadFilename_exists = False if loadFilename is None else os.path.exists(loadFilename)
    

    # Load model if a loadFilename is provided
    if loadFilename_exists:
        # If loading on same machine the model was trained on
        # checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        checkpoint = torch.load(loadFilename, map_location=torch.device(device=device))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']


    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, HIDDEN_SIZE)
    if loadFilename_exists:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(HIDDEN_SIZE, embedding, ENCODER_N_LAYERS, DROPOUT)
    decoder = LuongAttnDecoderRNN(ATTN_MODEL, embedding, HIDDEN_SIZE, voc.num_words, DECODER_N_LAYERS, DROPOUT)
    if loadFilename_exists:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')
    
    ### TRAIN ###########################################################################
    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LR)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LR * DECODER_LEARNING_RATIO)
    if loadFilename_exists:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # If you have cuda, configure cuda to call
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    # Run training iterations
    print("Starting Training!")
    trainIters(model_name=MODEL_NAME,
               voc=voc,
               pairs=pairs,
               encoder=encoder,
               decoder=decoder,
               encoder_optimizer=encoder_optimizer,
               decoder_optimizer=decoder_optimizer,
               embedding=embedding,
               encoder_n_layers=ENCODER_N_LAYERS,
               decoder_n_layers=DECODER_N_LAYERS,
               hidden_size=HIDDEN_SIZE,
               save_dir=SAVE_DIR,
               n_iteration=N_ITERATIONS,
               batch_size=BATCH_SIZE,
               print_every=PRINT_EVERY,
               save_every=SAVE_EVERY,
               clip=CLIPPING,
               corpus_name=DATA_NAME,
               loadFilename=loadFilename,
               device=device,
               max_length=MAX_LENGTH,
               teacher_forcing_ratio=TEACHER_FORCING_RATIO
               )
    
    ### EVALUATE MODEL ############################################################################
    if EVALUATE:
        # Set dropout layers to eval mode
        encoder.eval()
        decoder.eval()

        # Initialize search module
        searcher = GreedySearchDecoder(encoder, decoder)

        # Begin chatting (uncomment and run the following line to begin)
        evaluateInput(encoder, decoder, searcher, voc, MAX_LENGTH, device)