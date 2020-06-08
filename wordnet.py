from nltk.corpus import wordnet
import random
from random import shuffle
import re


random.seed(1)

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']

def get_only_chars(line):

	clean_line = ""

	line = line.replace("’", "")
	line = line.replace("'", "")
	line = line.replace("-", " ") #replace hyphens with spaces
	line = line.replace("\t", " ")
	line = line.replace("\n", " ")
	line = line.lower()

	for char in line:
		if char in 'qwertyuiopasdfghjklzxcvbnm ':
			clean_line += char
		else:
			clean_line += ' '

	clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
	
	if clean_line != '' and clean_line[0] == ' ':
		clean_line = clean_line[1:]
	return clean_line

def get_synonyms(word):
	synonyms = set()
	for syn in wordnet.synsets(word): 
		for l in syn.lemmas(): 
			synonym = l.name().replace("_", " ").replace("-", " ").lower()
			synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
			synonyms.add(synonym) 
	if word in synonyms:
		synonyms.remove(word)
	return list(synonyms)

def synonym_replacement(words, n):
	new_words = words.copy()
	random_word_list = list(set([word for word in words if word not in stop_words]))
	shuffle(random_word_list)
	num_replaced = 0
	for random_word in random_word_list:
		synonyms = get_synonyms(random_word)
		if len(synonyms) >= 1:
			synonym = random.choice(list(synonyms))
			new_words = [synonym if word == random_word else word for word in new_words]
			#print("replaced", random_word, "with", synonym)
			num_replaced += 1
		if num_replaced >= n: #only replace up to n words
			break

	#this is stupid but we need it, trust me
	sentence = ' '.join(new_words)
	new_words = sentence.split(' ')

	return new_words

def SR(sentence, alpha_sr, n_aug=9):

	sentence = get_only_chars(sentence)
	words = sentence.split(' ')
	num_words = len(words)

	augmented_sentences = []
	n_sr = max(1, int(alpha_sr*num_words))

	for _ in range(n_aug):
		a_words = synonym_replacement(words, n_sr)
		augmented_sentences.append(' '.join(a_words))

	augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
	shuffle(augmented_sentences)

	augmented_sentences.append(sentence)

	return augmented_sentences

def gen_sr_aug(train_orig, output_file, alpha_sr, n_aug):
	writer = open(output_file, 'w')
	lines = open(train_orig, 'r').readlines()
	for i, line in enumerate(lines):
		if i % 100 == 0:
			print(i)
		label = line[0]
		sentence = line[2:]
		aug_sentences = SR(sentence, alpha_sr=alpha_sr, n_aug=n_aug)
		for aug_sentence in aug_sentences:
			writer.write(label + "\t" + aug_sentence + '\n')
	writer.close()
	print("finished SR for", train_orig, "to", output_file, "with alpha", alpha_sr)
