import numpy as np
import matplotlib.pyplot as plt




# from http://www.lextek.com/manuals/onix/stopwords1.html


# note: an alternative source of stopwords
# from nltk.corpus import stopwords
# stopwords.words('english')

# load the reviews
# data courtesy of http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html



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



# create a word-to-index map so that we can create our word-frequency vectors later
# let's also save the tokenized versions so we don't have to tokenize again later

# now let's create our input matrices
# (N x D+1 matrix - keeping them together for now so we can shuffle more easily later


# shuffle the data and create train/test splits
# try it multiple times!


# let's look at the weights for each word
# try it with different threshold values!

# check misclassified examples

# since there are many, just print the "most" wrong samples

