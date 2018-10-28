"""
Constructing and loading dictionaries
"""

import pickle as pkl
import numpy
from collections import OrderedDict

def build_dictionary(text):
    """
    Build a dictionary
    text: list of sentences (pre-tokenized)
    """
    wordcount = OrderedDict()
    for cc in text:
        words = cc.split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 0
            wordcount[w] += 1
    words = list(wordcount.keys())
    freqs = list(wordcount.values())
    sorted_idx = numpy.argsort(freqs)[::-1]#每个词语出现频率的降序排列

    worddict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):#enumerate函数用于遍历序列中的元素以及它们的下标
        worddict[list(words)[sidx]] = idx+2# 0: <eos>, 1: <unk>
    return worddict, wordcount

# def load_dictionary(loc='/root/vseProj/vse/10crop_precomp_vocab.pkl'):
#     """
#     Load a dictionary
#     """
#     with open(loc, 'rb') as f:
#         worddict = pkl.load(f)
#     return worddict
#
# def save_dictionary(worddict, wordcount, loc):
#     """
#     Save a dictionary to the specified location
#     """
#     with open(loc, 'wb') as f:
#         pkl.dump(worddict, f)
#         pkl.dump(wordcount, f)


