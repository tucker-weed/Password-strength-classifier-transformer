import os
import numpy as np
import tensorflow as tf
from preprocess import *
from transformer_model import Transformer_Seq2Seq
import sys
import random
import pickle

# True: allows user to input passwords and check strength
# False: runs the test set
USER_INPUT = False

def train(model, train_french, train_english, eng_padding_index):
	"""
	Runs through one epoch - all training examples.

	:param model: the initialized model to use for forward and backward pass
	:param train_french: french train data (all data for training) of shape (num_sentences, 14)
	:param train_english: english train data (all data for training) of shape (num_sentences, 15)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:return: None
	"""

	# NOTE: For each training step, you should pass in the french sentences to be used by the encoder, 
	# and english sentences to be used by the decoder
	# - The english sentences passed to the decoder have the last token in the window removed:
	#	 [STOP CS147 is the best class. STOP *PAD*] --> [STOP CS147 is the best class. STOP] 
	# 
	# - When computing loss, the decoder labels should have the first word removed:
	#	 [STOP CS147 is the best class. STOP] --> [CS147 is the best class. STOP] 
    
	inds = np.arange(len(train_french))
	inds = tf.random.shuffle(inds)
	train_french = tf.gather(train_french, inds)
	train_english = tf.gather(train_english, inds)

	stopper = 0

	for i in range(0, len(train_french), model.batch_size):
		stopper += 1
		image = train_french[i:i + model.batch_size]
		label = train_english[i:i + model.batch_size]
		label = np.delete(label, -1, 1)
		label2 = np.delete(train_english[i:i + model.batch_size], 0, 1)
		mask = label2 != eng_padding_index

		with tf.GradientTape() as tape:
			preds = model([image, label])
			loss = model.loss_function(preds, label2, mask)
		if stopper % 10 == 0:
			acc = model.accuracy_function(preds, label2, mask)
			print("LOSS: {} | ACCURACY: {}".format(loss.numpy(), acc))

		gradients = tape.gradient(loss, model.trainable_variables)
		model.opt.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_french, test_english, eng_padding_index):
	"""
	Runs through one epoch - all testing examples.

	:param model: the initialized model to use for forward and backward pass
	:param test_french: french test data (all data for testing) of shape (num_sentences, 14)
	:param test_english: english test data (all data for testing) of shape (num_sentences, 15)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
	:returns: a tuple containing at index 0 the perplexity of the test set and at index 1 the per symbol accuracy on test set, 
	e.g. (my_perplexity, my_accuracy)
	"""

	# Note: Follow the same procedure as in train() to construct batches of data!
    
	loss_tracker = 0.0
	acc_tracker = 0.0
	total_words = 0

	for i in range(0, len(test_french), model.batch_size):
		image = test_french[i:i + model.batch_size]
		label = test_english[i:i + model.batch_size]
		label = np.delete(label, -1, 1)
		preds = model([image, label])

		label2 = np.delete(test_english[i:i + model.batch_size], 0, 1)
		mask = label2 != eng_padding_index
		batch_word_count = tf.cast(tf.reduce_sum(mask * 1), tf.float32)
		total_words += batch_word_count
		loss = model.loss_function(preds, label2, mask)
		acc = model.accuracy_function(preds, label2, mask)
		loss_tracker += loss
		acc_tracker += (acc * batch_word_count)

	return np.exp(loss_tracker / total_words), (acc_tracker / total_words)
	

def main():

	# Change this to False to run the test set rather than a prediction
	global USER_INPUT

	train_english, test_english, train_french, test_french, english_vocab, french_vocab, eng_padding_index = get_data('../data/data.csv')

	with open('eng.pkl', 'wb') as handle:
		pickle.dump(english_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open('fre.pkl', 'wb') as handle:
		pickle.dump(french_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

	model_args = (FRENCH_WINDOW_SIZE, len(french_vocab), ENGLISH_WINDOW_SIZE, len(english_vocab))
	model = Transformer_Seq2Seq(*model_args)
	
	train(model, train_french, train_english, eng_padding_index)

	model.save('./myModel')

if __name__ == '__main__':
	main()
