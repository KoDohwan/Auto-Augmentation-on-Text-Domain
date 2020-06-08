import random
import os
import re
from config import *

def clean_str(string):
    string = string.replace("-", " ")
    string = string.replace("\t", " ")
    string = string.replace("\n", " ")
    string = string.replace(" '", "'")
    string = string.replace("' ", " ")
    string = string.replace(" n't", "n't")
    string = re.sub(r"[^A-Za-z,!?\'\` ]", " ", string)
    string = re.sub(r" +", " ", string)
    string = string.lower()

    if string[0] == " ": 
        string = string[1:]

    return string

def data_clean():
    fout = open(all_path, 'w')
    with open(raw_path, encoding='latin-1') as f:
        for line in f:
            label = line[0]
            sen = line[2:]
            if sen == '\n': 
                continue

            sen = clean_str(sen)
            new = label + ' ' + sen + '\n'
            fout.write(new)
    fout.close()

def data_split():
    ftrain = open(train_path, 'w')
    ftest = open(test_path, 'w')

    text = open(all_path, 'r').readlines()
    test_index = -int(len(text) * 0.1)
    random.shuffle(text)
    test = text[test_index:]

    train = text[:test_index]
    train_index = int(len(train) * data_ratio)
    train = train[:train_index]
    
    ftrain.writelines(train)
    ftest.writelines(test)
    ftrain.close()
    ftest.close()

if __name__ == '__main__':
  
    # data_merge()
    data_clean()
    data_split()

    print('Finish')