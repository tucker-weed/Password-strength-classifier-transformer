import numpy as np
import tensorflow as tf
import csv

##########DO NOT CHANGE#####################
PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
FRENCH_WINDOW_SIZE = 14
ENGLISH_WINDOW_SIZE = 2
##########DO NOT CHANGE#####################

def pad_corpus(french, english):
	"""
	DO NOT CHANGE:

	arguments are lists of FRENCH, ENGLISH sentences. Returns [FRENCH-sents, ENGLISH-sents]. The
	text is given an initial "*STOP*".  All sentences are padded with "*STOP*" at
	the end.

	:param french: list of French sentences
	:param english: list of English sentences
	:return: A tuple of: (list of padded sentences for French, list of padded sentences for English)
	"""
	FRENCH_padded_sentences = []
	FRENCH_sentence_lengths = []
	for line in french:
		padded_FRENCH = line[:FRENCH_WINDOW_SIZE]
		padded_FRENCH += [STOP_TOKEN] + [PAD_TOKEN] * (FRENCH_WINDOW_SIZE - len(padded_FRENCH)-1)
		if len(padded_FRENCH) > FRENCH_WINDOW_SIZE:
			padded_FRENCH = padded_FRENCH[0 : FRENCH_WINDOW_SIZE]
		FRENCH_padded_sentences.append(padded_FRENCH)

	ENGLISH_padded_sentences = []
	ENGLISH_sentence_lengths = []
	for line in english:
		padded_ENGLISH = line[:ENGLISH_WINDOW_SIZE]
		padded_ENGLISH = [START_TOKEN] + padded_ENGLISH + [STOP_TOKEN] + [PAD_TOKEN] * (ENGLISH_WINDOW_SIZE - len(padded_ENGLISH)-1)
		if len(padded_ENGLISH) > ENGLISH_WINDOW_SIZE + 2:
			padded_ENGLISH = padded_ENGLISH[0 : ENGLISH_WINDOW_SIZE + 2]
		ENGLISH_padded_sentences.append(padded_ENGLISH)

	return FRENCH_padded_sentences, ENGLISH_padded_sentences

def build_vocab(sentences):
	"""
	DO NOT CHANGE

  Builds vocab from list of sentences

	:param sentences:  list of sentences, each a list of words
	:return: tuple of (dictionary: word --> unique index, pad_token_idx)
  """
	tokens = []
	for s in sentences: tokens.extend(s)
	all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + tokens)))

	vocab =  {word:i for i,word in enumerate(all_words)}

	return vocab,vocab[PAD_TOKEN]

def convert_to_id(vocab, sentences):
	"""
	DO NOT CHANGE

    Convert sentences to indexed 

	:param vocab:  dictionary, word --> unique index
	:param sentences:  list of lists of words, each representing padded sentence
	:return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
    """
	return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])


def read_data(file_name):
	"""
	DO NOT CHANGE

  Load text data from file

	:param file_name:  string, name of data file
	:return: list of sentences, each a list of words split on whitespace
  """
	text = []
	labels = []
	with open(file_name, 'rt') as csv_file:
		reader = csv.reader(csv_file, delimiter=',')
		flag = False
		for line in reader:
			if not flag:
				flag = True
				continue
			text.append([char for char in line[0]])
			labels.append([line[1]])
	return labels, text
	

def get_data(data_file):
	"""
	Use the helper functions in this file to read and parse training and test data, then pad the corpus.
	Then vectorize your train and test data based on your vocabulary dictionaries.

	:param french_training_file: Path to the french training file.
	:param english_training_file: Path to the english training file.
	:param french_test_file: Path to the french test file.
	:param english_test_file: Path to the english test file.
	
	:return: Tuple of train containing:
	(2-d list or array with english training sentences in vectorized/id form [num_sentences x 15] ),
	(2-d list or array with english test sentences in vectorized/id form [num_sentences x 15]),
	(2-d list or array with french training sentences in vectorized/id form [num_sentences x 14]),
	(2-d list or array with french test sentences in vectorized/id form [num_sentences x 14]),
	english vocab (Dict containg word->index mapping),
	french vocab (Dict containg word->index mapping),
	english padding ID (the ID used for *PAD* in the English vocab. This will be used for masking loss)
	"""
	# MAKE SURE YOU RETURN SOMETHING IN THIS PARTICULAR ORDER: train_english, test_english, train_french, test_french, english_vocab, french_vocab, eng_padding_index
	
	#1) Read English and French Data for training and testing (see read_data)

	english1, french1 = read_data(data_file)
	english2 = english1[round(len(english1) * 0.85) : ]
	english1 = english1[0 : round(len(english1) * 0.85)]
	french2 = french1[round(len(french1) * 0.85) : ]
	french1 = french1[0 : round(len(french1) * 0.85)]
	

	#2) Pad training data (see pad_corpus)

	french1, english1 = pad_corpus(french1, english1)

	#3) Pad testing data (see pad_corpus)

	french2, english2 = pad_corpus(french2, english2)

	#4) Build vocab for french (see build_vocab)

	fredict, _ = build_vocab(french1)

	#5) Build vocab for english (see build_vocab)

	engdict, tok = build_vocab(english1)

	#6) Convert training and testing english sentences to list of IDS (see convert_to_id)

	#7) Convert training and testing french sentences to list of IDS (see convert_to_id)

	english1 = convert_to_id(engdict, english1)
	english2 = convert_to_id(engdict, english2)
	french1 = convert_to_id(fredict, french1)
	french2 = convert_to_id(fredict, french2)

	return english1, english2, french1, french2, engdict, fredict, tok

