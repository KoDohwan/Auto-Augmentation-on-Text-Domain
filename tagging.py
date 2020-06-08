#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
from torchtext import data
import operator
import nltk
import operation
import requests
import itertools
from config import *
from methods import *
from nltk.parse.corenlp import CoreNLPParser as CNP
parser = CNP(url='http://localhost:9000')


# In[2]:


def traverse_tree(tree, word_path):
    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            traverse_tree(subtree, word_path + ' ' + (str(subtree).split()[0][1:]))
        else:
            # print(subtree, word_path)
            path_list.append((subtree, word_path.split()))
    
    return path_list


d1_set = {'NN', 'JJ', 'RB', 'VB'}
d2_set = {'AD', 'NP', 'VP', 'PP', 'WH'}
position_set = set()
for i, j in itertools.product(d2_set, d1_set):
    position_set.add(i + j)


# In[3]:


word_path = ''
path_list = []
examples = []
augment_examples = []

res = open(tag_path, 'w')

with open(train_path) as f:
    for i, line in enumerate(f):
        label = line[0]
        text = line[2:]
        if i % 100 == 0:
            print(i)

        path_list = []
        try:
            parsed_sen = next(parser.raw_parse(text))
        except requests.exceptions.HTTPError:
            print(str(i + 1) + ' : pass sentence')
            continue
        except StopIteration:
            print(str(i + 1) + ' : StopIteration')
            continue

        path_list = traverse_tree(parsed_sen, word_path)
    
        tag_dict = {}
        for p in position_set:
            tag_dict[p] = []
        
        for words in path_list:
            length = len(words[1])
            d1 = words[1][length - 1][:2]
            d2 = words[1][length - 2][:2]
            position = d2 + d1

            if position in position_set:
                tag_dict[position].append(words[0])
                
        res.write(str((line, tag_dict)) + '\n')

res.close()
print('Done tagging!!!')

