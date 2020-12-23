import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

class Transformer_Seq2Seq(tf.keras.Model):
	def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):

		######vvv DO NOT CHANGE vvv##################
		super(Transformer_Seq2Seq, self).__init__()

		self.french_vocab_size = french_vocab_size # The size of the french vocab
		self.english_vocab_size = english_vocab_size # The size of the english vocab

		self.french_window_size = french_window_size # The french window size
		self.english_window_size = english_window_size # The english window size
		######^^^ DO NOT CHANGE ^^^##################


		# TODO:
		# 1) Define any hyperparameters
		# 2) Define embeddings, encoder, decoder, and feed forward layers

		# Define batch size and optimizer/learning rate
		self.learning_rate = 0.001
		self.batch_size = 1000
		self.embedding_size = 66
		self.opt = tf.optimizers.Adam(learning_rate=self.learning_rate)

		# Define english and french embedding layers:

		self.EE = tf.keras.layers.Embedding(self.english_vocab_size, self.embedding_size, embeddings_initializer='uniform')
		self.EF = tf.keras.layers.Embedding(self.french_vocab_size, self.embedding_size, embeddings_initializer='uniform')
		
		# Create positional encoder layers

		self.PEF = transformer.Position_Encoding_Layer(self.french_window_size, self.embedding_size)
		self.PEE = transformer.Position_Encoding_Layer(self.english_window_size - 1, self.embedding_size)

		# Define encoder and decoder layers:
	
		# Define dense layer(s)

		self.en1 = transformer.Transformer_Block(self.embedding_size, is_decoder=False, multi_headed=True)
		self.en2 = transformer.Transformer_Block(self.embedding_size, is_decoder=False, multi_headed=True)
		self.en3 = transformer.Transformer_Block(self.embedding_size, is_decoder=False, multi_headed=True)
		self.de1 = transformer.Transformer_Block(self.embedding_size, is_decoder=True, multi_headed=True)
		self.de2 = transformer.Transformer_Block(self.embedding_size, is_decoder=True, multi_headed=True)
		self.de3 = transformer.Transformer_Block(self.embedding_size, is_decoder=True, multi_headed=True)
	
		# Define dense layer(s)
		self.Dsoft = tf.keras.layers.Dense(self.english_vocab_size, activation='softmax')

	@tf.function
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to french sentences
		:param decoder_input: batched ids corresponding to english sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""
	
		# TODO:
		#1) Add the positional embeddings to french sentence embeddings
		#2) Pass the french sentence embeddings to the encoder
		#3) Add positional embeddings to the english sentence embeddings
		#4) Pass the english embeddings and output of your encoder, to the decoder
		#5) Apply dense layer(s) to the decoder out to generate probabilities
	
		outF = self.EF(encoder_input)
		poF = self.PEF(outF)
		out1 = self.en1(poF)
		out1 = self.en2(out1)
		out1 = self.en3(out1)

		outE = self.EE(decoder_input)
		poE = self.PEE(outE)
		out2 = self.de1(poE, out1)
		out2 = self.de2(out2, out1)
		out2 = self.de3(out2, out1)
    
		prbs = self.Dsoft(out2)

		return prbs

	def accuracy_function(self, prbs, labels, mask):
		"""
		DO NOT CHANGE

		Computes the batch accuracy
		
		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: scalar tensor of accuracy of the batch between 0 and 1
		"""

		decoded_symbols = tf.argmax(input=prbs, axis=2)
		accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32), mask))
		return max(0.0, (accuracy - 0.5) / 0.5)


	def loss_function(self, prbs, labels, mask):
		"""
		Calculates the model cross-entropy loss after one forward pass
		Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""

		return tf.reduce_sum(tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(labels, prbs), mask))		

	def __call__(self, *args, **kwargs):
		return super(Transformer_Seq2Seq, self).__call__(*args, **kwargs)