# This code is for my NLP Udemy class, which can be found at:
# https://deeplearningcourses.com/c/data-science-natural-language-processing-in-python
# https://www.udemy.com/data-science-natural-language-processing-in-python
# It is written in such a way that tells a story.
# i.e. So you can follow a thought process of starting from a
# simple idea, hitting an obstacle, overcoming it, etc.
# i.e. It is not optimized for anything.

# Author: http://lazyprogrammer.me

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import nltk

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LinearRegression
from bs4 import BeautifulSoup

# puts word in base form
wordnet_lemmatizer = WordNetLemmatizer()


# from http://www.lextek.com/manuals/onix/stopwords1.html
stopwords = set(w.rstrip() for w in open('data/stopwords.txt'))

# note: an alternative source of stopwords
# from nltk.corpus import stopwords
# stopwords.words('english')

# load the reviews
# data courtesy of http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
positive_reviews = BeautifulSoup(open('data/l38/positive.review').read(), features="html5lib")
positive_reviews = positive_reviews.find_all('review_text')

negative_reviews = BeautifulSoup(open('data/l38/negative.review').read(), features="html5lib")
negative_reviews = positive_reviews.find_all('review_text')


# first let's just try to tokenize the text using nltk's tokenizer
# let's take the first review for example:
# t = positive_reviews[0]
# nltk.tokenize.word_tokenize(t.text)
#
# notice how it doesn't downcase, so It != it
# not only that, but do we really want to include the word "it" anyway?
# you can imagine it wouldn't be any more common in a positive review than a negative review
# so it might only add noise to our model.
# so let's create a function that does all this pre-processing for us

def my_tokenizer(s):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t)>2] # remove short words, they're probably not useful
    tokens =  [wordnet_lemmatizer.lemmatize(t) for t in tokens ] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    return tokens


# create a word-to-index map so that we can create our word-frequency vectors later
# let's also save the tokenized versions so we don't have to tokenize again later
word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []
original_reviews = []

for review in positive_reviews:
    tokens = my_tokenizer(review)
    positive_tokenized.append(tokens)
    original_reviews.append(review)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token]=current_index
            current_index+=1

for review in negative_reviews:
    tokens = my_tokenizer(review)
    negative_tokenized.append(tokens)
    original_reviews.append(review)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token]=current_index
            current_index+=1

print("len(word_index_map)", len(word_index_map))

# now let's create our input matrices
# (N x D+1 matrix - keeping them together for now so we can shuffle more easily later


# shuffle the data and create train/test splits
# try it multiple times!


# let's look at the weights for each word
# try it with different threshold values!

# check misclassified examples

# since there are many, just print the "most" wrong samples

