#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 23:42:48 2018

@author: charlie
"""
from gensim.test.utils import get_tmpfile
from gensim.models import word2vec, KeyedVectors
from gensim import corpora
import logging
import os.path


logging.basicConfig( format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

cur_dir = os.getcwd()
word_vectors_fname = cur_dir+ '/model_vectors/word_vectors.kv'
if os.path.exists(word_vectors_fname):
    word_vectors = KeyedVectors.load(word_vectors_fname, mmap='r')
else:
    sentences = word2vec.Text8Corpus("text/text8")
    dict_tokens = corpora.Dictionary(sentences)
    dict_tokens.save('/model_vectors/dict_tokens.dict')
    print(dict_tokens)
#    print(dict_tokens.token2id)
    model = word2vec.Word2Vec(sentences, min_count=10, iter=5, size=50, workers=4)
    word_vectors_path = get_tmpfile(word_vectors_fname)
    word_vectors = model.wv
    word_vectors.save(word_vectors_path)
    # If training is done, only want to query the vector space, then we can delete the model instance and switch to the kv instance
    del model

# Save the vector space of words 




 # word_vectors['men'] & model.wv['men']
vec_men = word_vectors['men']
vec_man = word_vectors['man']
vec_women = word_vectors['women']
vec_woman = word_vectors['woman']
vec_frog = word_vectors['frog']

#cos_simi_men_man = 1 - spatial.distance.cosine(vec_men, vec_man)
cos_simi_men_man = word_vectors.similarity('men', 'man')
cos_simi_men_women = word_vectors.similarity('men', 'women')
cos_simi_men_frog = word_vectors.similarity('men', 'frog')
#model = word2vec

# prediction
# model.most_similar(positive=['women', 'king'], negative=['men'], topn=1)
res1 = word_vectors.most_similar(positive=['women', 'king'], negative=['men'], topn=1)
res2 = word_vectors.most_similar(positive=['women', 'king'], negative=['man'], topn=1)
res3 = word_vectors.most_similar(['man'])
res4 = word_vectors.most_similar(['men'])

print("\nMost similar word of positive=['women', 'king'], negative=['men'] is ")
print(res1)
print("\nMost similar word of positive=positive=['women', 'king'], negative=['man'] is ")
print(res2)
print("\nMost similar word of ['man'] is ")
print(res3)
print("\nMost similar word of ['men'] is ")
print(res4)

# prediction
print("\n new predictions:")
pred_list = ["he is she", "big bigger bad", "going went being"]
for sentence in pred_list:
    a, b, x = sentence.split()
    pred_word = word_vectors.most_similar([x, b], [a])[0][0]
    print("%s is to %s as %s is to %s" % (a, b, x, pred_word))







