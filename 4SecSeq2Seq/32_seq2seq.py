# original file wseq2seq.py
# data from http://www.manythings.org/anki/

from __future__ import print_function, division
from builtins import range, input

import os, sys
import  numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense, Embedding, Input, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical

try:
    import keras.backend as K
    if len(K.tensorflow_backend._get_available_gpus()) > 0:
        from keras.layers import CuDNNLSTM as LSTM
        from keras.layers import CuDNNGRU as GRU
except:
    pass

# config
BATCH_SIZE = 64 # BATCH SIZE FOR TRAINING
EPOCHS = 40
LATENT_DIM = 256
NUM_SAMPLES = 10000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100


# where we store the data
input_texts = [] # sentence in original language
target_texts = [] # sentence in target language
target_texts_inputs = [] # sentence in target language offset by 1(for teacher forcing, using eos and sos)

# load the data
# data from http://www.manythings.org/anki/
t = 0
for line in open('../large_files/ukr.txt', encoding='UTF-8'):
    # only keep a limited number of samples
    t += 1
    if t> NUM_SAMPLES:
        break

    # input and target are separated by tab
    if '\t' not in line:
        continue

    # split up the input and translation
    input_text, translation, *rest = line.rstrip().split('\t')

    # make the target input and output
    # recall will be teacher forcing
    target_text = translation + ' <eos>'
    target_texts_input = '<sos>' + translation

    input_texts.append(input_text)
    target_texts.append(target_texts_input)
print("num samples:", len(input_texts))

# tokenize the inputs
tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

# get the word to index mapping for input language
word2idx_inputs = tokenizer_inputs.word_index
print("Found @ unique input tokens(words)" % len(word2idx_inputs))

# determine maximum length input sentence
max_len_input = max(len(s) for s in input_sequences)

