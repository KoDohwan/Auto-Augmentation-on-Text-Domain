# import plaidml.keras
# plaidml.keras.install_backend()

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
import keras.layers as layers
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import math
import time
import numpy as np
import random
from random import randint
random.seed(3)
import datetime, re, operator
from random import shuffle
from time import gmtime, strftime
from ast import literal_eval
import gc

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #get rid of warnings
from os import listdir
from os.path import isfile, join, isdir
import pickle
from config import *

#loading a pickle file
def load_pickle(file):
	return pickle.load(open(file, 'rb'))

#create an output folder if it does not already exist
def confirm_output_folder(output_folder):
	if not os.path.exists(output_folder):
	    os.makedirs(output_folder)

#get full image paths
def get_txt_paths(folder):
    txt_paths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f)) and '.txt' in f]
    if join(folder, '.DS_Store') in txt_paths:
        txt_paths.remove(join(folder, '.DS_Store'))
    txt_paths = sorted(txt_paths)
    return txt_paths

#get subfolders
def get_subfolder_paths(folder):
    subfolder_paths = [join(folder, f) for f in listdir(folder) if (isdir(join(folder, f)) and '.DS_Store' not in f)]
    if join(folder, '.DS_Store') in subfolder_paths:
        subfolder_paths.remove(join(folder, '.DS_Store'))
    subfolder_paths = sorted(subfolder_paths)
    return subfolder_paths

#get all image paths
def get_all_txt_paths(master_folder):

    all_paths = []
    subfolders = get_subfolder_paths(master_folder)
    if len(subfolders) > 1:
        for subfolder in subfolders:
            all_paths += get_txt_paths(subfolder)
    else:
        all_paths = get_txt_paths(master_folder)
    return all_paths

#get the pickle file for the glove so you don't have to load the entire huge file each time
def gen_vocab_dicts(folder, output_pickle_path, huge_glove):

    vocab = set()
    text_embeddings = open(huge_glove, 'r').readlines()
    glove = {}

    #get all the vocab
    all_txt_paths = get_all_txt_paths(folder)
    print(all_txt_paths)

    #loop through each text file
    for txt_path in all_txt_paths:

    	# get all the words
    	try:
    		all_lines = open(txt_path, "r").readlines()
    		for line in all_lines:
    			words = line[:-1].split(' ')
    			for word in words:
    			    vocab.add(word)
    	except:
    		print(txt_path, "has an error")
    
    print(len(vocab), "unique words found")

    # load the word embeddings, and only add the word to the dictionary if we need it
    for line in text_embeddings:
        items = line.split(' ')
        word = items[0]
        if word in vocab:
            vec = items[1:]
            glove[word] = np.asarray(vec, dtype = 'float32')
    print(len(glove), "matches between unique words and glove dictionary")
        
    pickle.dump(glove, open(output_pickle_path, 'wb'))
    print("dictionaries outputted to", output_pickle_path)

#getting the x and y inputs in numpy array form from the text file
def train_x_y(train_txt, num_classes, glove_len, input_size, glove, percent_dataset):
    #read in lines
    train_lines = open(train_txt, 'r').readlines()
    shuffle(train_lines)
    train_lines = train_lines[:int(percent_dataset*len(train_lines))]
    num_lines = len(train_lines)
    
    temp = []
    tag_list = []
    for line in train_lines:
        line = literal_eval(line)
        temp.append(line[0])
        tag_list.append(line[1])
    train_lines = temp
    
    #initialize x and y matrix
    x_matrix = None
    y_matrix = None

    try:
        x_matrix = np.zeros((num_lines, input_size, glove_len))
    except:
        print("Error!", num_lines, input_size, glove_len)
    y_matrix = np.zeros((num_lines, num_classes))

    #insert values
    for i, line in enumerate(train_lines):

        label = int(line[0])
        sentence = line[2:]

        #insert x
        words = sentence.split(' ')
        words = words[:x_matrix.shape[1]] #cut off if too long
        for j, word in enumerate(words):
            if word in glove:
                x_matrix[i, j, :] = glove[word]

        #insert y
        y_matrix[i][label] = 1.0
    
    return x_matrix, y_matrix, tag_list

def get_tag(train_txt, percent_dataset):
    train_lines = open(train_txt, 'r').readlines()
    shuffle(train_lines)
    train_lines = train_lines[:int(percent_dataset*len(train_lines))]
    num_lines = len(train_lines)
    
    
    tag_list = []
    sentences = []
    y_matrix = np.zeros((num_lines, num_classes))
    for i, line in enumerate(train_lines):
        line = literal_eval(line)
        label = int(line[0][0])
        
        sentence = line[0][2:]
        sentences.append(sentence)
        
        tag = line[1]
        tag_list.append(tag)
        y_matrix[i][label] = 1.0

    return tag_list, sentences, y_matrix

def get_x(sentences, input_size, glove_len, glove):
    num_lines = len(sentences)
    try:
        x_matrix = np.zeros((num_lines, input_size, glove_len))
    except:
        print("Error!", num_lines, input_size, glove_len)

    for i, sentence in enumerate(sentences):
        words = sentence.split(' ')
        words = words[:x_matrix.shape[1]] 
        for j, word in enumerate(words):
            if word in glove:
                x_matrix[i, j, :] = glove[word]
                
    return x_matrix

def get_x_y(train_txt, num_classes, glove_len, input_size, glove, percent_dataset):
    #read in lines
    train_lines = open(train_txt, 'r').readlines()
    shuffle(train_lines)
    train_lines = train_lines[:int(percent_dataset*len(train_lines))]
    num_lines = len(train_lines)

    #initialize x and y matrix
    x_matrix = None
    y_matrix = None

    try:
        x_matrix = np.zeros((num_lines, input_size, glove_len))
    except:
        print("Error!", num_lines, input_size, glove_len)
    y_matrix = np.zeros((num_lines, num_classes))

    #insert values
    for i, line in enumerate(train_lines):

        label = int(line[0])
        sentence = line[2:]

        #insert x
        words = sentence.split(' ')
        words = words[:x_matrix.shape[1]] 
        for j, word in enumerate(words):
            if word in glove:
                x_matrix[i, j, :] = glove[word]

        #insert y
        y_matrix[i][label] = 1.0

    return x_matrix, y_matrix

#building the model in keras
def rnn(sentence_length, glove_len, num_classes):
	model = None
	model = Sequential()
	model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(sentence_length, glove_len)))
	model.add(Dropout(0.5))
	model.add(Bidirectional(LSTM(32, return_sequences=False)))
	model.add(Dropout(0.5))
	model.add(Dense(20, activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	#print(model.summary())
	return model

#building the cnn in keras
def cnn(sentence_length, glove_len, num_classes):
    adam = Adam(lr = 0.001)
    model = None
    model = Sequential()
    model.add(layers.Conv1D(128, 5, activation='relu', input_shape=(sentence_length, glove_len)))
    model.add(layers.GlobalMaxPooling1D())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

#one hot to categorical
def one_hot_to_categorical(y):
    assert len(y.shape) == 2
    return np.argmax(y, axis=1)

def get_now_str():
    return str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))