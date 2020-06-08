import wordnet
import random
from config import *
from ast import literal_eval

def ADNN(tag_sen):
    new_sen = tag_sen[0]
    words = tag_sen[1]['ADNN']
    for word in words:
        if len(wordnet.get_synonyms(word)) == 0:
            continue
        new_word = random.sample(wordnet.get_synonyms(word), 1)
        new_sen = new_sen.replace(word, new_word[0])
    return new_sen

def ADJJ(tag_sen):
    new_sen = tag_sen[0]
    words = tag_sen[1]['ADJJ']
    for word in words:
        if len(wordnet.get_synonyms(word)) == 0:
            continue
        new_word = random.sample(wordnet.get_synonyms(word), 1)
        new_sen = new_sen.replace(word, new_word[0])
    return new_sen

def ADRB(tag_sen):
    new_sen = tag_sen[0]
    words = tag_sen[1]['ADRB']
    for word in words:
        if len(wordnet.get_synonyms(word)) == 0:
            continue
        new_word = random.sample(wordnet.get_synonyms(word), 1)
        new_sen = new_sen.replace(word, new_word[0])
    return new_sen

def ADVB(tag_sen):
    new_sen = tag_sen[0]
    words = tag_sen[1]['ADVB']
    for word in words:
        if len(wordnet.get_synonyms(word)) == 0:
            continue
        new_word = random.sample(wordnet.get_synonyms(word), 1)
        new_sen = new_sen.replace(word, new_word[0])
    return new_sen

def NPNN(tag_sen):
    new_sen = tag_sen[0]
    words = tag_sen[1]['NPNN']
    for word in words:
        if len(wordnet.get_synonyms(word)) == 0:
            continue
        new_word = random.sample(wordnet.get_synonyms(word), 1)
        new_sen = new_sen.replace(word, new_word[0])
    return new_sen

def NPJJ(tag_sen):
    new_sen = tag_sen[0]
    words = tag_sen[1]['NPJJ']
    for word in words:
        if len(wordnet.get_synonyms(word)) == 0:
            continue
        new_word = random.sample(wordnet.get_synonyms(word), 1)
        new_sen = new_sen.replace(word, new_word[0])
    return new_sen

def NPRB(tag_sen):
    new_sen = tag_sen[0]
    words = tag_sen[1]['NPRB']
    for word in words:
        if len(wordnet.get_synonyms(word)) == 0:
            continue
        new_word = random.sample(wordnet.get_synonyms(word), 1)
        new_sen = new_sen.replace(word, new_word[0])
    return new_sen

def NPVB(tag_sen):
    new_sen = tag_sen[0]
    words = tag_sen[1]['NPVB']
    for word in words:
        if len(wordnet.get_synonyms(word)) == 0:
            continue
        new_word = random.sample(wordnet.get_synonyms(word), 1)
        new_sen = new_sen.replace(word, new_word[0])
    return new_sen

def VPNN(tag_sen):
    new_sen = tag_sen[0]
    words = tag_sen[1]['VPNN']
    for word in words:
        if len(wordnet.get_synonyms(word)) == 0:
            continue
        new_word = random.sample(wordnet.get_synonyms(word), 1)
        new_sen = new_sen.replace(word, new_word[0])
    return new_sen

def VPJJ(tag_sen):
    new_sen = tag_sen[0]
    words = tag_sen[1]['VPJJ']
    for word in words:
        if len(wordnet.get_synonyms(word)) == 0:
            continue
        new_word = random.sample(wordnet.get_synonyms(word), 1)
        new_sen = new_sen.replace(word, new_word[0])
    return new_sen

def VPRB(tag_sen):
    new_sen = tag_sen[0]
    words = tag_sen[1]['VPRB']
    for word in words:
        if len(wordnet.get_synonyms(word)) == 0:
            continue
        new_word = random.sample(wordnet.get_synonyms(word), 1)
        new_sen = new_sen.replace(word, new_word[0])
    return new_sen

def VPVB(tag_sen):
    new_sen = tag_sen[0]
    words = tag_sen[1]['VPVB']
    for word in words:
        if len(wordnet.get_synonyms(word)) == 0:
            continue
        new_word = random.sample(wordnet.get_synonyms(word), 1)
        new_sen = new_sen.replace(word, new_word[0])
    return new_sen

def PPNN(tag_sen):
    new_sen = tag_sen[0]
    words = tag_sen[1]['PPNN']
    for word in words:
        if len(wordnet.get_synonyms(word)) == 0:
            continue
        new_word = random.sample(wordnet.get_synonyms(word), 1)
        new_sen = new_sen.replace(word, new_word[0])
    return new_sen

def PPJJ(tag_sen):
    new_sen = tag_sen[0]
    words = tag_sen[1]['PPJJ']
    for word in words:
        if len(wordnet.get_synonyms(word)) == 0:
            continue
        new_word = random.sample(wordnet.get_synonyms(word), 1)
        new_sen = new_sen.replace(word, new_word[0])
    return new_sen

def PPRB(tag_sen):
    new_sen = tag_sen[0]
    words = tag_sen[1]['PPRB']
    for word in words:
        if len(wordnet.get_synonyms(word)) == 0:
            continue
        new_word = random.sample(wordnet.get_synonyms(word), 1)
        new_sen = new_sen.replace(word, new_word[0])
    return new_sen

def PPVB(tag_sen):
    new_sen = tag_sen[0]
    words = tag_sen[1]['PPVB']
    for word in words:
        if len(wordnet.get_synonyms(word)) == 0:
            continue
        new_word = random.sample(wordnet.get_synonyms(word), 1)
        new_sen = new_sen.replace(word, new_word[0])
    return new_sen

def WHNN(tag_sen):
    new_sen = tag_sen[0]
    words = tag_sen[1]['WHNN']
    for word in words:
        if len(wordnet.get_synonyms(word)) == 0:
            continue
        new_word = random.sample(wordnet.get_synonyms(word), 1)
        new_sen = new_sen.replace(word, new_word[0])
    return new_sen

def WHJJ(tag_sen):
    new_sen = tag_sen[0]
    words = tag_sen[1]['WHJJ']
    for word in words:
        if len(wordnet.get_synonyms(word)) == 0:
            continue
        new_word = random.sample(wordnet.get_synonyms(word), 1)
        new_sen = new_sen.replace(word, new_word[0])
    return new_sen

def WHRB(tag_sen):
    new_sen = tag_sen[0]
    words = tag_sen[1]['WHRB']
    for word in words:
        if len(wordnet.get_synonyms(word)) == 0:
            continue
        new_word = random.sample(wordnet.get_synonyms(word), 1)
        new_sen = new_sen.replace(word, new_word[0])
    return new_sen

def WHVB(tag_sen):
    new_sen = tag_sen[0]
    words = tag_sen[1]['WHVB']
    for word in words:
        if len(wordnet.get_synonyms(word)) == 0:
            continue
        new_word = random.sample(wordnet.get_synonyms(word), 1)
        new_sen = new_sen.replace(word, new_word[0])
    return new_sen

def get_transformations():
    return [ADNN, ADJJ, ADRB, ADVB, NPNN, NPJJ, NPRB, NPVB, VPNN, VPJJ, VPRB, VPVB, PPNN, PPJJ, PPRB, PPVB, WHNN, WHJJ, WHRB, WHVB]

if __name__ == '__main__':
    transformations = get_transformations()

    with open(tag_path, 'r') as f:
        for i, line in enumerate(f):
            if i > 10:
                break

            line = literal_eval(line)
            label = line[0][0]
            tag = line[1]
            sen2 = sen1 = line[0][2:]

            
            for t in transformations:
                sen2 = t((sen2, tag))

            print(label)
            print(sen1)
            print(sen2)