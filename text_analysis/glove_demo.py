#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 12:45:30 2018

@author: charlie
"""

import itertools
from gensim.models.word2vec import Text8Corpus
from glove import Corpus, Glove
import os

cur_dir = os.getcwd()
glove_fname = '/glove.model'
corpus_fname = "/corpus.model"
if os.path.exists(cur_dir + glove_fname):
    glove = Glove.load(cur_dir+glove_fname)
#    corpus = Corpus.load(cur_dir+corpus_fname)
else:
    sentences = list(itertools.islice(Text8Corpus('text/text8'), None))
    corpus = Corpus()
    corpus.fit(sentences, window = 10)
    
    glove = Glove(no_components=100, learning_rate = 0.05)
    glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    
    glove.save(cur_dir + glove_fname)
    corpus.save(cur_dir+corpus_fname)

glove.most_similar('men') # Parameters are hashable string not list
glove.word_vectors[glove.dictionary['perfect']]