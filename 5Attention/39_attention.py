from __future__ import print_function, division
from builtins import range, input

import os, sys

from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding, \
    Bidirectional, RepeatVector, Concatenate, Activation, Dot, Lambda
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import keras.backend as K

import numpy as np
import matplotlib.pyplot as plt

try:
    import keras.backend as K
    if len(K.tensorflow_backend._get_available_gpus()) > 0:
        from keras.layers import CuDNNLSTM as LSTM
        from keras.layers import CuDNNGRU as GRU
except:
    pass

# make sure we do softmax over the time axis
# expected shape is N x T x D
# note: the latest version of keras allows you to pass the axis arg
def softmax_over_time(x):
    assert(K.ndim(x) > 2)
    e = K.exp(x-K.max(x, axis=1, keepdims=True))
    s = K.sum(e, axis=1, keepdims=True)
    return e/s

# config
BATCH_SIZE = 64
EPOCHS = 30
LATENT_DIM = 400
LATENT_DIM_DECODER  =400 # IDEA - make it different to ensure all things fit together properly!
NUM_SAMPLES = 20000
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS =20000
EMBEDDING_DIM = 100


# Where we store the data
input_texts = [] # sentences in original language
target_texts = [] # sentences in target language
target_texts_inputs = [] # sentences in targhet language offset by 1

# load the data
t = 0
for line in open('../large_files/ukr.txt', encoding='UTF-8'):
    # only keep limited number of samples
    t += 1
    if t > NUM_SAMPLES:
        break

    # inputs and targets are separated by tab
    if '\t' not in line:
        continue

    # split up the input and translation
    input_text, translation, *rest = line.rstrip().split('\t')

    # make the target input and putput
    # recall we'll be using teacher forcing
    target_text = translation + ' <eos>'
    target_text_input ='<sos> ' + translation

    input_texts.append(input_text)
    target_texts.append(target_text)
    target_texts_inputs.append(target_text_input)
print("Num samples:", len(input_texts))


# tokenize the inputs
tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

# get the word to index mapping for input language
word2idx_inputs = tokenizer_inputs.word_index
print("Found %s unique index tokens" % len(word2idx_inputs))

# determine maximum length output sequence
max_len_input = max(len(s) for s in input_sequences)

# tokenize the outputs
# dont filter any special characters
# otherwise <sos> and <oes> won't appear
tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs)
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)

# get word to index mapping for output language
word2idx_outputs = tokenizer_outputs.word_index
print("Found %s unique index tokens" % len(word2idx_outputs))

# store number of output words for later
# remember to add 1 as index starts from 1
num_words_output = len(word2idx_outputs) + 1

# determine maximum length output sequence
max_len_target = max(len(s) for s in target_sequences)

# pad the sequences
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)
print("encoder_inputs.shape:", encoder_inputs.shape)
print("encoder_inputs[0]:", encoder_inputs[0])

encoder_outputs = pad_sequences(target_sequences, maxlen=max_len_target)
print("encoder_outputs.shape:", encoder_outputs.shape)
print("encoder_outputs[0]:", encoder_outputs[0])

