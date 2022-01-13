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
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    'two_supporting_fact_10k' : 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-fact_{}.txt',
}


def tokenize(sent):
    '''
    Return the tokens of sentence, including punctuation.

    >>> tokenize("Bob dropped the apple. Where is the apple?")
    ['Bob',  'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
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
        if(nid == 1):
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
    train_stories = get_stories(tar.extract(challenge.format('train')))
    test_stories = get_stories(tar.extract(challenge.format('test')))

    # group all stories together
    stories = train_stories + test_stories

    # so we can get max length of each story, of each sentence of each question
    story_maxlen = max((len(s) for x, _, _ in stories for s in x))
    story_max_sents = max(len(x) for x, _, _ in stories)
    query_maxlen = max(len(x) for _,x, _ in stories)

    # Create vocabulary of corpus and find size
    vocab = sorted(set(flatten(stories)))
    vocab.instert(0, '<PAD>')
    vocab_size = len(vocab)

    # Create an index mapping for the vocabulary
    word2idx = {w : i for i, w in enumerate(vocab)}

    inputs_train, queries_train, answers_train = vectorize_stories(train_stories, word2idx, story_maxlen, query_maxlen)
    inputs_test, queries_test, answers_test = vectorize_stories(test_stories, word2idx, story_maxlen, query_maxlen)

    # convert all into 3D numpy arrays
    input_train = stack_inputs(inputs_train, story_max_sents, story_maxlen)
    test_train = stack_inputs(inputs_test, story_max_sents, story_maxlen)
    print("inputs_train.shape, inputs_test.shape", inputs_train.shape, inputs_test.shape)

    # return model inputs for keras
    return train_stories, test_stories, \
        inputs_train, queries_train, answers_train, \
        inputs_test, queries_test, answers_test, \
        story_max_sents, story_maxlen, query_maxlen, \
        vocab, vocab_size
