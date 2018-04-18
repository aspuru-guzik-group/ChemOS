#!/usr/bin/env python

# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random




if __name__ == '__main__':
	import json 
	with open('intents.json') as json_data:
		intents = json.load(json_data)

	# organize our document
	words = []
	classes = []
	documents = []
	ignore_words = ['?']

	# loop through each sentence
	for intent in intents['intents']:
		for pattern in intent['patterns']:
			# tokenize each word 
			w = nltk.word_tokenize(pattern)
			# add word to word list
			words.extend(w)
			# add to document
			documents.append((w, intent['tag']))
			# add to classes list
			if intent['tag'] not in classes:
				classes.append(intent['tag'])

	# massage words
	words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
	words = sorted(list(set(words)))
	
	# remove duplicates
	classes = sorted(list(set(classes)))


	# create our training data
	training = []
	output = []
	output_empty = [0] * len(classes)

	for doc in documents:
		bag = []
		pattern_words = doc[0]
		pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
		for w in words:
			bag.append(1) if w in pattern_words else bag.append(0)

		output_row = list(output_empty)
		output_row[classes.index(doc[1])] = 1

		training.append([bag, output_row])

	random.shuffle(training)
	training = np.array(training)

	# create train and test lists
	train_x = list(training[:, 0])
	train_y = list(training[:, 1])

	# now build the training model
	tf.reset_default_graph()
	net = tflearn.input_data(shape = [None, len(train_x[0])])
	net = tflearn.fully_connected(net, 8)
	net = tflearn.fully_connected(net, 8)
	net = tflearn.fully_connected(net, len(train_y[0]), activation = 'softmax')
	net = tflearn.regression(net)

	# define model and setup
	model = tflearn.DNN(net, tensorboard_dir = 'tflearn_logs')
	# start training 
	model.fit(train_x, train_y, n_epoch = 1000, batch_size = 8, show_metric = True)
	model.save('ChatBotModel/model.tflearn')

	# save everything
	import pickle
	pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open('training_data', 'wb'))	
	
