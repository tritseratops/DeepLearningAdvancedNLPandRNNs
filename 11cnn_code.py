from __future__ import print_function, division
from builtins import range
#you may need to update your version of future
#pip install -U future

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences # make sequences of same length
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from sklearn.metrics import roc_auc_score


# download the large_files:
# https://lazyprogrammer.me/course_files/toxic_comment_train.csv
# http://nlp.stanford.edu/data/glove.6B.zip

#some configuration
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 1 #10

print('Loading word vectors...')
# print(os.system("dir"))
word2vec = {}
with open(os.path.join('large_files/glove.6B.%sd.txt' % EMBEDDING_DIM), encoding="utf8") as f:
    # is just space-separated text file in the format:
    #word vec[0] vec[1] vec[2]
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))

#prepare text samples and their labels
print("Loading in comments...")

train = pd.read_csv("large_files/toxic_comment_train.csv")
sentences = train["comment_text"].fillna("DUMMY_VALUE").values
possible_labels = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
targets = train[possible_labels].values

print("max sequence length:", max(len(s) for s  in sentences))
print("max sequence length:", min(len(s) for s  in sentences))

s =  sorted(len(s) for s in sentences)

print("median sequence length:", s[len(s) // 2])


# convert the sensentes (strings) into integers
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
# print("Sequences:", sequences)
print("Sequences qty:", len(sequences))

#get word -> integer mapping
word2idx = tokenizer.word_index
print("Found %s unique tokens." % len(word2idx))

# pad sequences so that we get a N x T matrix
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print("Shape of data tensor:", data.shape)

#prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE, len(word2idx)+1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            #words not found in embedding index will be all zeros
            embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into embedding layer
# note that we set trainable = False so as to keep embeddings fixed
embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=MAX_SEQUENCE_LENGTH,
    trainable=False
)


print("Building model...")

#train a 1D convnet with global maxpooling
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
x = Conv1D(128,3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128,3, activation='relu')(x)
x = MaxPooling1D(3)(x)
x = Conv1D(128,3, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(len(possible_labels), activation='sigmoid')(x)

model = Model(input_, output)
model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

print('Training model....')
r = model.fit(
    data,
    targets,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT
)

#plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

print("r.history type:"+str(type(r.history)))
print("r.history keys:"+str(r.history.keys()))
# accuracies
plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

# plot the mean AUC over each label
p = model.predict(data)
aucs = []
for j in range(6):
    auc = roc_auc_score(targets[:,j], p[:,j])
print(np.mean(aucs))


# second model
embedding_layer2 = Embedding(
    num_words,
    EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=MAX_SEQUENCE_LENGTH,
    trainable=False
)
#train a 1D convnet with global maxpooling
x2 = embedding_layer2(input_)
x2 = Conv1D(128,3, activation='relu')(x2)
x2 = MaxPooling1D(3)(x2)
x2 = Conv1D(128,3, activation='relu')(x2)
x2 = MaxPooling1D(3)(x2)
x2 = Conv1D(128,3, activation='relu')(x2)
x2 = GlobalMaxPooling1D()(x2)
x2 = Dense(128, activation='relu')(x2)
output2 = Dense(len(possible_labels), activation='sigmoid')(x2)


model_same_input = Model(input_, output2)
model_same_input.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

model_same_input_output = Model(input_, output)
model_same_input_output.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

# forth model
embedding_layer3 = Embedding(
    num_words,
    EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=MAX_SEQUENCE_LENGTH,
    trainable=False
)

input2 = Input(shape=(MAX_SEQUENCE_LENGTH,))
x3 = embedding_layer3(input2)
x3 = Conv1D(128,3, activation='relu')(x3)
x3 = MaxPooling1D(3)(x3)
x3 = Conv1D(128,3, activation='relu')(x3)
x3 = MaxPooling1D(3)(x3)
x3 = Conv1D(128,3, activation='relu')(x3)
x3 = GlobalMaxPooling1D()(x3)
x3 = Dense(128, activation='relu')(x3)
output3 = Dense(len(possible_labels), activation='sigmoid')(x3)
model_diff_input_output = Model(input2, output3)
model_diff_input_output.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

print("model.get_weight(s", model.get_weights())
print("model.get_weight()", model.get_weights().shape)
print("model.get_weights()[0]", model.get_weights()[0])
print("model.get_weights()[1]", model.get_weights()[1])
print("model.get_weights()[0].head", model.get_weights()[0].head)
print("model.get_weights()[1].head", model.get_weights()[1].head)
print("model_same_input.get_weight()s", model_same_input.get_weights())
print("model_same_input_output.get_weight()s", model_same_input_output.get_weights())