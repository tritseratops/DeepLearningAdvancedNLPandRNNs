from __future__ import print_function, division
from builtins import range, input

import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
import re
import tarfile

from keras.models import Model
from keras.layers import Dense, Embedding, Input, Lambda, Reshape, add, dot, Activation
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.utils.data_utils import get_file

# get the data and open compressed file using tarfile library
# https://research.fb.com/downloads/babi
path = get_file(
    'babi-tasks-v1-2.tar.gz',
    origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz'
)
tar = tarfile.open(path)

#relevant data in the tar file
# there is lots more dat in there, check it out
challenges = {
    # QA1 with 10000 samples
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa6_yes-no-questions_{}.txt',
    'two_supporting_facts_10k' : 'tasks_1-20_v1-2/en-10k/qa6_yes-no-questions_{}.txt',
}


def tokenize(sent):
    '''
    Return the tokens of sentence, including punctuation.

    >>> tokenize("Bob dropped the apple. Where is the apple?")
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''

    return [x.strip() for x in re.split('(\W+?)', sent) if x.strip()]

def get_stories(f):
    # data will return a list of triples
    # each triple contains:
    # 1. a story
    # 2. a question about the story
    # 3. answer to the question
    data = []

    # use this list to keep track of the story so far
    story = []

    # print a random story, helpful to see the data
    printed = False
    for line  in f:
        line = line.decode('utf-8').strip()

        # split the line number from the rest of the line
        nid, line = line.split(' ', 1)

        # check if we should begin a new story
        if(int(nid) == 1):
            story = []

        # this line contains a question and answer if it has a tab
        # question<TAB>answer
        # it also tells us which line in the story is relevant to the answer
        # Note we actually ignore this fact, since model should learn what lines is important
        # Note: The max line number is not the number of line in the story
        #   since lines with question does not contain any story
        # one story may contain multiple questions
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)

            # numbering each line is very useful
            # it's the equivalent of adding a unique token to the front
            # of each sentence
            story_so_far = [[str(i)] + s for i, s in enumerate(story) if s]

            # uncomment if you want to see what a story looks like
            # if not printed and np.random.rand() < 0.5:
            #     print("story so far:", story_so_far)
            #     printed = True
            data.append((story_so_far, q, a))
            story.append('')
        else:
            # just add the line to the current story
            story.append(tokenize(line))
    return data

# challenge = challenges['single_supporting_fact_10k']
# train_stories = get_stories(tar.extractfile(challenge.format('train')))
# test_stories = get_stories(tar.extractfile(challenge.format('test')))

# recursively flatten: a list, to get a list of words
def should_flatten(el):
    return not isinstance(el, (str, bytes))

def flatten (l):
    for el in l:
        if should_flatten(el):
            yield from flatten(el)
        else:
            yield el

# convert stories from words into lists of words indexes (integers)
# pad each sequence so they are of the same length
# we will need to repad the story later, so that each story is of the same length
def vectorize_stories(data, word2idx, story_maxlen, query_maxlen):
    inputs, queries, answers = [], [], []
    for story, query, answer in data:
        inputs.append([[word2idx[w] for w in sentence] for sentence in story])
        queries.append([word2idx[w] for w in query])
        answers.append([word2idx[answer]])
    return(
            [pad_sequences(x, maxlen=story_maxlen) for x in inputs],
            pad_sequences(queries, maxlen=query_maxlen),
            np.asarray(answers)
    )

# this is like 'pad_sequences' but for entire stories
# we are padding each story with zeroes for every story
# so it had the same number of sentences
# append an array of zeroes of size:
# (max_sentences - num sentences in the story, max words in sentence)
def stack_inputs(inputs, story_maxsents, story_maxlen):
    for i, story in enumerate(inputs):
        inputs[i] = np.concatenate(
            [
                story,
                np.zeros((story_maxsents - story.shape[0], story_maxlen), 'int')
            ]
        )
    return np.stack(inputs)


# function to get data
# since we want to load single supporting fact data
# and two supporting fact data later
def get_data(challenge_type):
    # input should either be single supporting fact 'single_supporting_fact_10k' or 'two_supporting_fact_10k'
    challenge = challenges[challenge_type]

    # returns a list of triples of:
    # (story, question, answer)
    # story in list of sentences
    # question in a sentence
    # answer is a word
    train_stories = get_stories(tar.extractfile(challenge.format('train')))
    test_stories = get_stories(tar.extractfile(challenge.format('test')))

    # group all stories together
    stories = train_stories + test_stories

    # so we can get max length of each story, of each sentence of each question
    story_maxlen = max((len(s) for x, _, _ in stories for s in x))
    story_max_sents = max(len(x) for x, _, _ in stories)
    query_maxlen = max(len(x) for _,x, _ in stories)

    # Create vocabulary of corpus and find size
    vocab = sorted(set(flatten(stories)))
    vocab.insert(0, '<PAD>')
    vocab_size = len(vocab)

    # Create an index mapping for the vocabulary
    word2idx = {w : i for i, w in enumerate(vocab)}

    inputs_train, queries_train, answers_train = vectorize_stories(train_stories, word2idx, story_maxlen, query_maxlen)
    inputs_test, queries_test, answers_test = vectorize_stories(test_stories, word2idx, story_maxlen, query_maxlen)

    # convert all into 3D numpy arrays
    inputs_train = stack_inputs(inputs_train, story_max_sents, story_maxlen)
    inputs_test = stack_inputs(inputs_test, story_max_sents, story_maxlen)
    print("inputs_train.shape, inputs_test.shape", inputs_train.shape, inputs_test.shape)

    # return model inputs for keras
    return train_stories, test_stories, \
        inputs_train, queries_train, answers_train, \
        inputs_test, queries_test, answers_test, \
        story_max_sents, story_maxlen, query_maxlen, \
        vocab, vocab_size


# get the single supporting fact data
train_stories, test_stories, \
    inputs_train, queries_train, answers_train, \
    inputs_test, queries_test, answers_test, \
    story_maxsents, story_maxlen, query_maxlen, \
    vocab, vocab_size = get_data('single_supporting_fact_10k')


#### create the model ####
embedding_dim = 15


# turn the story into sequence of embedding vectors
# one for each storyline
# treating each story line as a bag of words
input_story = Input((story_maxsents, story_maxlen))
embedded_story = Embedding(vocab_size, embedding_dim)(input_story)
embedded_story = Lambda(lambda x: K.sum(x, axis=2))(embedded_story)
print("input_story.shape, embedded_story.shape", input_story.shape, embedded_story.shape)

# turn the question into embedding
# also bag of words
input_question = Input((query_maxlen,))
embedded_question = Embedding(vocab_size, embedding_dim)(input_question)
embedded_question = Lambda(lambda x: K.sum(x, axis=1))(embedded_question)

# add a sequence length  of 1 so that it can be dotted with story later
embedded_question = Reshape((1, embedding_dim))(embedded_question)
print("input_question.shape, embedded_question.shape:", input_question.shape, embedded_question.shape)

# calculate weights of each storyline
# embedded_story.shape = (N, num sentences, embedding_dim)
# embedded question.shape = (N, 1, embedding dim)
x = dot([embedded_story, embedded_question], 2)
x = Reshape((story_maxsents,))(x) # flatten the vector
x = Activation('softmax')(x)
story_weights = Reshape((story_maxsents, 1))(x) # unflatten it again to be dotted later
print("story_weights.shape", story_weights.shape)

x= dot([story_weights, embedded_story], 1)
x = Reshape((embedding_dim,))(x)
ans = Dense(vocab_size, activation='softmax')(x)

# make the model
model = Model([input_story, input_question], ans)

# compile the models
model.compile(
    optimizer=RMSprop(learning_rate=1e-2),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train the model
r = model.fit(
    [inputs_train, queries_train],
    answers_train,
    epochs=4,
    batch_size=32,
    validation_data=([inputs_test, queries_test], answers_test)
)

# Check how we weight each input sequence given a story and question
debug_model=  Model([input_story, input_question], story_weights)

print("model.summary():", model.summary())
print("debug_model.summary():", debug_model.summary())

# choose a random story
story_idx = np.random.choice(len(train_stories))
i =inputs_train[story_idx:story_idx+1]
q = queries_train[story_idx:story_idx+1]
w = debug_model.predict([i,q]).flatten()

story, question, ans = train_stories[story_idx]
print("Story:", story_idx)
print("\n")
for i, line in enumerate(story):
    print("{:1.5f}".format(w[i]), "\t", ".".join(line))

print("question:", " ".join(question))
print("answer:", ans)

# pause so we cab see the output
input("Hit enter to continue\n\n")




# two supporting facts
train_stories, test_stories, \
    inputs_train, queries_train, answers_train, \
    inputs_test, queries_test, answers_test, \
    story_maxsents, story_maxlen, query_maxlen, \
    vocab, vocab_size = get_data('two_supporting_facts_10k')


#### Create the model #####
embedding_dim = 30

# make a function for embed and layer so we can reuse it
def emded_and_sum(x, axis=2):
    x = Embedding(vocab_size, embedding_dim)(x)
    x = Lambda(lambda x: K.sum(x, axis))(x)
    return x


input_story = Input((story_maxsents, story_maxlen))
input_question = Input((query_maxlen,))


# embed the inputs
embedded_story = emded_and_sum(input_story)
embedded_question = emded_and_sum(input_question,1)

# final dense will be used for each hop
dense_layer = Dense(embedding_dim, activation='elu')


# define one hop
# the query can be the question or the answer to the previous hop
def hop(query, story):
    # query,shape = (embedding_dim,)
    # story.shape = (num_sentences, embedding_dim)
    x = Reshape ((1, embedding_dim))(query) # make it (1, embedding dim)
    x = dot([story, x], 2)
    x =Reshape((story_maxsents,))(x) # flaten for softmax
    x = Activation('softmax')(x)
    story_weights = Reshape((story_maxsents,1 ))(x) # unflatten for dotting

    # makes a new embedding
    story_embedding2 = emded_and_sum(input_story)
    x = dot([story_weights, story_embedding2], 1)
    x = Reshape((embedding_dim,))(x)
    x = dense_layer(x)
    return x, story_embedding2, story_weights

ans1, embedded_story, story_weights1 = hop(embedded_question, embedded_story)
ans2, _ ,             story_weights2 = hop(ans1, embedded_story)

# get the final answer
ans = Dense(vocab_size, activation='softmax')(ans2)

# build the model
model2 = Model([input_story, input_question], ans)

# compile the model
model2.compile(
    optimizer=RMSprop(learning_rate=5e-3),
    loss='sparse_categorical_crossentropy',
    metrics= ['accuracy']

)

# fit the model
r = model2.fit(
    [inputs_train, queries_train],
    answers_train,
    batch_size=32,
    epochs=100,
    validation_data=([inputs_test, queries_test], answers_test)
)


### print story line weights again
debug_model2 = Model(
    [input_story, input_question],
    [story_weights1, story_weights2]
)


# chose a random story
story_idx = np.random.choice(len(train_stories))

# get weigths from debug model
i = inputs_train[story_idx:story_idx+1]
q = queries_train[story_idx:story_idx+1]
w1, w2 = debug_model2.predict([i,q])
w1 = w1.flatten()
w2 = w2.flatten()

story, question, ans = train_stories[story_idx]
print("story:\n")
for j, line in enumerate(story):
    print("{:1.5f}".format(w1[j]), "\t", "{:1.5f}".format(w2[j]), "\t", " ".join(line))

print("question:", " ".join(question))
print("answer:", ans)
print("prediction:", vocab[np.argmax(model2.predict([i,q])[0])])

while True:
    # chose a random story
    story_idx = np.random.choice(len(train_stories))

    # get weigths from debug model
    i = inputs_train[story_idx:story_idx + 1]
    q = queries_train[story_idx:story_idx + 1]
    w1, w2 = debug_model2.predict([i, q])
    w1 = w1.flatten()
    w2 = w2.flatten()

    story, question, ans = train_stories[story_idx]
    print("story:\n")
    for j, line in enumerate(story):
        print("{:1.5f}".format(w1[j]), "\t", "{:1.5f}".format(w2[j]), "\t", " ".join(line))

    print("question:", " ".join(question))
    print("answer:", ans)
    print("prediction:", vocab[np.argmax(model2.predict([i, q])[0])])
    q = input("Do you want more examples?(y/n)")
    if q.lower() == "n":
        break

# plot some data
plt.plot(r.history['loss'], label="loss")
plt.plot(r.history['val_loss'], label="val_loss")
plt.legend()
plt.show()

# plot some data
plt.plot(r.history['accuracy'], label="accuracy")
plt.plot(r.history['val_accuracy'], label="val_acc")
plt.legend()
plt.show()
