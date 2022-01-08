# original file wseq2seq.py
# data from http://www.manythings.org/anki/

from __future__ import print_function, division
from builtins import range, input

import os, sys

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt

try:
    import keras.backend as K

    if len(K.tensorflow_backend._get_available_gpus()) > 0:
        from keras.layers import CuDNNLSTM as LSTM
        from keras.layers import CuDNNGRU as GRU
except:
    pass

# config
BATCH_SIZE = 64  # BATCH SIZE FOR TRAINING
EPOCHS =  1 #40
LATENT_DIM = 256
NUM_SAMPLES = 10000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

# where we store the data
input_texts = []  # sentence in original language
target_texts = []  # sentence in target language
target_texts_inputs = []  # sentence in target language offset by 1(for teacher forcing, using eos and sos)

# load the data
# data from http://www.manythings.org/anki/
t = 0
for line in open('../large_files/ukr.txt', encoding='UTF-8'):  # , encoding='UTF-8'
    # only keep a limited number of samples
    t += 1
    if t > NUM_SAMPLES:
        break

    # input and target are separated by tab
    if '\t' not in line:
        continue

    # split up the input and translation
    input_text, translation, *rest = line.rstrip().split('\t')

    # make the target input and output
    # recall will be teacher forcing
    target_text = translation + ' <eos>'
    target_text_input = '<sos> ' + translation

    input_texts.append(input_text)
    target_texts.append(target_text)
    target_texts_inputs.append(target_text_input)
print("num samples:", len(input_texts))

# tokenize the inputs
tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts) # sentence translations original - eng 10000  encoded

# get the word to index mapping for input language
word2idx_inputs = tokenizer_inputs.word_index
print("Found %s unique input tokens(words)" % len(word2idx_inputs))

# determine maximum length input sentence
max_len_input = max(len(s) for s in input_sequences) # max number oif words in sentence

# tokenize outputs
# don't filter out special characters
# otherwsise <sos> and <eos> dont appear
tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
print("len(target_texts)", len(target_texts))
print("len(target_texts_inputs)", len(target_texts_inputs))
print("target_texts[0:10]", target_texts[0:10])
print("target_texts_inputs[0:10]", target_texts_inputs[0:10])
tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs)  # inefficient - why?
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts) # sentence translations - ukr 10000 encoded
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs) # both original and translation encoded [10000 x sentence length] both encoded
print("target_sequences_inputs[0:10]", target_sequences_inputs[0:10])

# get the word to index mapping for output language
word2idx_outputs = tokenizer_outputs.word_index
print("Found %s unique output tokens(words)" % len(word2idx_outputs))

# store number of output words for later
# add 1 as index starts at 1
num_words_output = len(word2idx_outputs) + 1

# maximum output sentence length
max_len_target = max(len(s) for s in target_sequences)

# pad the sequences
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input) # 10000 x 5
print("Encoder input shape:", encoder_inputs.shape)
print("encoded_inputs[0]:", encoder_inputs[0])

print("len(target_input_sequences):", len(target_sequences_inputs))
print("target_input_sequences[0:10]:", target_sequences_inputs[0:10])
decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=max_len_target)  # 10000 x 10
print("Decoder input shape:", decoder_inputs.shape)
print("Decoded_inputs[0]:", decoder_inputs[0])

decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')

# store all pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
with open(os.path.join('../large_files/glove.6B.%sd.txt' % EMBEDDING_DIM), encoding='UTF-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec

print('Found %s word vectors.' % len(word2vec))

# prepare embedding matrix
print("Filling pre trained embeddings ...")
num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM)) # 1951 x 100
for word, i in word2idx_inputs.items():
    if i < MAX_NUM_WORDS:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
# print("embedding_matrix.shape", embedding_matrix.shape)

# create embedding layer
encoder_embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=max_len_input,
    # trainable = True
)
print("num_words:",num_words) # 1951
print("EMBEDDING_DIM:",EMBEDDING_DIM) # 100
print("embedding_matrix.shape:",embedding_matrix.shape) # 1951x100
print("embedding_layer:", encoder_embedding_layer) #  <keras.layers.embeddings.Embedding object at 0x000001D85B04C2B0>
print("input_length:",max_len_input) #5

# create targets since we cannot use sparse categorical entropy when we have sequences
decoder_targets_one_hot = np.zeros(  # 10000 x 10 x 6361
    (
        len(input_texts),
        max_len_target,
        num_words_output
    ),
    dtype='float32'
)

# assign the values
# per translation sentence x word in sentence x word in pre-trained loaded vector, if word is in translation we put 1
# otherwise 0
for i, d in enumerate(decoder_targets):
    for t, word in enumerate(d):
        if word != 0:
            decoder_targets_one_hot[i, t, word] = 1

##### Build the model
encoder_inputs_placeholder = Input(shape=(max_len_input,))
x = encoder_embedding_layer(encoder_inputs_placeholder)
print("x", x) #
print("x.shape", x.shape) #

encoder = LSTM(
    LATENT_DIM,
    return_state=True,
    # droupout = 0.5 - now available for me?
)
encoder_outputs, h, c = encoder(x)

print("encoder", encoder) #
# print("encoder.shape", encoder.shape) #
# encoder_outputs,h = encoder(x)  # - when GRU
print("encoder_outputs", encoder_outputs) #  KerasTensor(type_spec=TensorSpec(shape=(None, 256)
print("encoder_outputs.shape", encoder_outputs.shape) # (None, 256)

# keep only states to pass into decoder
encoder_states = [h, c]
# encoder_state = [h] # for GRU
print("h:", h)
print("c:", c)

# set up the decoder , using [h, c] as initial state
decoder_inputs_placeholder = Input(shape=(max_len_target,))

# this word embedding will not use pretrained vectors
# although you could- TODO
decoder_embedding = Embedding(num_words_output, EMBEDDING_DIM)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)
print("num_words_output", num_words_output) # 6361
print("EMBEDDING_DIM", EMBEDDING_DIM) # 100
print("decoder_embedding", decoder_embedding) # input shape(None, 10), output shape(None, 10, 100)
# print("decoder_embedding.shape", decoder_embedding.shape) #
print("decoder_inputs_x", decoder_inputs_x) # erasTensor(type_spec=TensorSpec(shape=(None, 10, 100)
print("decoder_inputs_x.shape", decoder_inputs_x.shape) # (None, 10, 100)


# since decoder is "to-many" model we want to have return_sequences = True
decoder_lstm = LSTM(
    LATENT_DIM,
    return_sequences=True,
    return_state=True,
    # dropout=0.5 # dropout not available on GPU
)

decoder_outputs, _, _ = decoder_lstm(
    decoder_inputs_x,
    initial_state=encoder_states
)
print("decoder_lstm", decoder_lstm) #
# print("decoder_lstm.shape", decoder_lstm.shape) # ERROR
print("decoder_outputs", decoder_outputs) # KerasTensor(type_spec=TensorSpec(shape=(None, 10, 256)
print("decoder_outputs.shape", decoder_outputs.shape) # (None, 10, 256)

# for gru
# decoder_outputs, _ = decoder_gru(
#     decoder_inputs_x,
#     initial_state=encoder_states
# )

# final dense layer for predictions
decoder_dense = Dense(num_words_output, activation='softmax')
print("decoder_dense", decoder_dense) # no useful info
# print("decoder_dense.shape", decoder_dense.shape) # error

decoder_outputs = decoder_dense(decoder_outputs)

# create the model object
model = Model([encoder_inputs_placeholder, decoder_inputs_placeholder],
              decoder_outputs)
print("encoder_inputs_placeholder", encoder_inputs_placeholder) # tensor x5
print("decoder_inputs_placeholder", decoder_inputs_placeholder) # tensor x10
print("decoder_outputs", decoder_outputs) # tensor x10x6361

def custom_loss(y_true, y_pred):
    # both are of shape N x T x K
    mask = K.cast(y_true > 0, dtype='float32')
    out = mask * y_true * K.log(y_pred)
    return -K.sum(out) / K.sum(mask)


def acc(y_true, y_pred):
    # both are of shape N x T x K
    targ = K.argmax(y_true, axis=-1)
    pred = K.argmax(y_pred, axis=-1)
    correct = K.cast(K.equal(targ, pred), dtype='float32')

    # 0 is padding, dont include those
    mask = K.cast(K.greater(targ, 0), dtype='float32')
    n_correct = K.sum(mask * correct)
    n_total = K.sum(mask)
    return n_correct / n_total

model.compile(optimizer='adam', loss=custom_loss, metrics=[acc])

# Compile the model and train it  MB for categorical crossentropy?
# model.compile(
#     optimizer='rmsprop',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )
print("encoder_inputs.shape", encoder_inputs.shape) # 10000 x 5
print("decoder_inputs.shape", decoder_inputs.shape)  # 10000 x 10
print("decoder_targets_one_hot.shape", decoder_targets_one_hot.shape) # 10000 x 10 x 6361

# X = [encoder_inputs, decoder_inputs]
# print("X.shape", X.shape)

r = model.fit(
    [encoder_inputs, decoder_inputs], decoder_targets_one_hot,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
)

# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()

# Save model
model.save('s2s.h5')

print("model.get_weights()", model.get_weights())
print("type(model.get_weights())", type(model.get_weights()))
print("model.get_weights().shape", model.get_weights().shape)
exit()


# Make predictions
#  in order to make predictions we have to make another model
# that can take RNN state and previous word as input
# and accept T=1 sequence

# The encoder woul;d be standalone
# from this we will get out initial decoder hidden state
encoder_model = Model(encoder_inputs_placeholder, encoder_states)

decoder_state_input_h = Input(shape=(LATENT_DIM,))
decoder_state_input_c = Input(shape=(LATENT_DIM,))
decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

# this time we want to kleep decoder states too, to be output
# by our sampling model
decoder_outputs, h, c  =  decoder_lstm(
    decoder_inputs_single_x,
    initial_state = decoder_state_inputs
)


decoder_states = [h, c]

decoder_outputs = decoder_dense(decoder_outputs)

# thew sampling model\
# inputs: y(t-1), h(t-1), c(t-1)
# outputs y(t), h(t), c(t)
decoder_model = Model(
    [decoder_inputs_single] + decoder_state_inputs,
    [decoder_outputs] + decoder_states
)

# map indexes back to real words
idx2wrd_eng = {v:k for k,v in word2idx_inputs.items()}
idx2word_ukr = {v:k for k,v in word2idx_outputs.items()}

# translate sentence from sentence vectors
def decode_sequence(input_seq):
    # encode the inputs as state vectors - create thought vector
    states_value = encoder_model.predict(input_seq)

    # generate empty target sequence of length 1
    target_seq = np.zeros((1,1))

    # populate the first character of target sequence with start character
    # Note: tokenizer lower-cases all words
    target_seq[0, 0] = word2idx_outputs['<sos>']

    # if we get this we break
    eos = word2idx_outputs['<eos>']

    # create the translation
    output_sentence = []
    for _ in range (max_len_target):
        output_tokens, h,c = decoder_model.predict(
            [target_seq] + states_value
        )

        # Get next word
        idx = np.argmax(output_tokens[0, 0, :])

        # end sentence if eos
        if eos == idx:
            break

        word = ''
        if idx>0:
            # print('len(idx2word_ukr):', len(idx2word_ukr))
            # print('idx2word_ukr:', idx2word_ukr)
            word = idx2word_ukr[idx]
            output_sentence.append(word)

        # update encoder input
        # which is the word just generated
        target_seq[0, 0] = idx

        # update states
        states_value = [h, c]

    return ' '.join(output_sentence)


while True:
    i = np.random.choice(len(input_texts))
    input_seq = encoder_inputs[i:i+1]
    translation = decode_sequence(input_seq)

    print('-')
    print('Input:', input_texts[i])
    print('Translation:', translation)

    ans = input('Continue? [Y/n]')
    if ans and ans.lower().startswith('n'):
        break