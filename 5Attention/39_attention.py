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

decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=max_len_target, padding='post')
print("decoder_inputs.shape:", decoder_inputs.shape)
print("decoder_inputs[0]:", decoder_inputs[0])

decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')

# store all pre-trained word vectors
print("Loading word vectors....")
word2vec = {}
with open(os.path.join('../large_files/glove.6B.%sd.txt' % EMBEDDING_DIM), encoding='UTF-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec

print('Found %s word vectors' % len(word2vec))

# prepare embedding matrix
num_words_input = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
embedding_matrix = np.zeros((num_words_input, EMBEDDING_DIM))
for word, word_idx in word2idx_inputs.items():
    if word_idx < MAX_NUM_WORDS:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            # all not found words would be zeroes
            embedding_matrix[word_idx] = embedding_vector


# create embedding layer
encoder_embeding = Embedding(
    num_words_input, # input dim
    EMBEDDING_DIM, # output dim
    weights=[embedding_matrix],
    input_length=max_len_input, # max original sentence length
    # trainable = True
)

# create targets since we can't use sparse categorical cross entropy when we have sequences
num_inputs = len(input_sequences) # originalk sentences
decoder_one_hot_targets = np.zeros((num_inputs, # number of original sentences in translations - same as number of translations
                                    max_len_target, # maximum translation sentence length
                                    num_words_output) # number of words in translation vocabulary
                                   , dtype='float32')

# we use decoder target because those are already sequenced and padded translations
for target_sentence_idx, target_sentence_sequence in enumerate(decoder_targets):
    for target_word_order_idx, word_code in enumerate(target_sentence_sequence):
        if word_code > 0:
            decoder_one_hot_targets[target_sentence_idx,target_word_order_idx,word_code] = 1


# build the model

# setup the encoder
encoder_input_placeholder = Input(shape=(max_len_input,))
print("max_len_input:", max_len_input)
x = encoder_embeding(encoder_input_placeholder)
encoder_declaration = Bidirectional(LSTM(
    LATENT_DIM, # we play with it I believe to achieve better results
    return_sequences=True,
    # dropout = 0.5
))
encoder_outputs = encoder_declaration(x)

# set up the decoder
decoder_input_placeholder = Input(shape=max_len_target) # actually decoder get thought vector
# -so why we have here max_len_target  -which is length of translated sentence?

# this word embedding would not use pretrained ectors, although we could
decoder_embedding = Embedding(num_words_output, EMBEDDING_DIM)
decoder_input_x = decoder_embedding(decoder_input_placeholder)



###### Attentionc ###
# attention layers have to be global because
# they will be repeated Ty times at the decoder
attn_repeat_layer = RepeatVector(max_len_input)
attn_concat_layer = Concatenate(axis=-1)
attn_dense1 = Dense(10, activation='tanh')
attn_dense2 = Dense(1, activation=softmax_over_time)
attn_dot = Dot(axes=1) # perform the weighted sum of alpha[t] * h[t]

def one_step_attention(h, st_1):
    # h = h(1), ..., h(Tx), shape = (Tx, LATENT_DIM * 2)
    # st_1 = s(t-1), shape = (LATENT_DIM_DECODER,)

    # copy s(t-1) Tx times
    # now shape = (Tx, LATENT_DIM_DECODER)
    st_1 = attn_repeat_layer(st_1)

    # Concatenate all h(t)'s with (st-1)
    # Now of shape (Tx, LATENT_DIM_DECODER + LATENT_DIM *2)
    x = attn_concat_layer([h, st_1])

    # Neural Newt first Layer
    x = attn_dense1(x)

    # Neural Net second layer with special softmax overtime
    alphas = attn_dense2(x)

    # "Dot" the alpha's and the h's
    # Remember a.dot(b) = sum over a[t] * b[t]
    context = attn_dot([alphas, h])

    return context

# define rest of the decoder (after attention)
decoder_lstm = LSTM(LATENT_DIM_DECODER, return_state=True)
decoder_dense = Dense(num_words_output, activation='softmax')

initial_s = Input(shape=(LATENT_DIM_DECODER,), name='s0')
initial_c = Input(shape=(LATENT_DIM_DECODER,), name='c0')
context_last_word_concat_layer = Concatenate(axis=2)

# unlike previous seq2seq we can't get output in one step
# instead we need to do Ty steps
# and in each step we need to consider all Tx's

# s,c weill be reassigned each iteration of the loop
s =initial_s
c = initial_c

# collect outputs in a list at first
outputs = []
for t in range(max_len_target): # Ty times
    # get the context using attention
    context = one_step_attention(encoder_outputs,s) # here we created specific NN for each Ty, encoder-outputs
    # is a declaration of encoder NN

    # we need a different layer for each time step
    selector = Lambda(lambda  x: x[:, t:t+1]) # so for each input we take all first dim and 1 cell defined in loop head
    # goping by Ty and take all data for Ty
    xt = selector(decoder_input_x) # decoder_input_x is embedding layer num_words_output=10, EMBEDDING_DIM=400

    # combine
    decoder_lstm_input = context_last_word_concat_layer([context, xt])

    # pass the combined [context, last word] into LSTM
    # along with s, c
    # get the new [s, c] and output
    o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[s, c])

    print("o:", o)
    # final dense layer to get new word prediction
    decoder_outputs = decoder_dense(o)
    outputs.append(decoder_outputs)






