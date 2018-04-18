#!/usr/bin/env python 

import os, random
import numpy as np
import json, pickle
import nltk
from nltk.stem.lancaster import LancasterStemmer

import tensorflow as tf
import tflearn


#============================================================

class Bot(object):
	
	ERROR_THRESHOLD = 0.2	

	def __init__(self):
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
		self.stemmer = LancasterStemmer()
		self._load_bot()
		self._load_model()


	def _load_bot(self):
		try:
			self.bot_data = pickle.load(open('ChatBot/training_data', 'rb'))
		except:
			self.bot_data = pickle.load(open('Communicator/ChatBot/training_data', 'rb'))
		self.words   = self.bot_data['words']
		self.classes = self.bot_data['classes']
		self.train_x = self.bot_data['train_x']
		self.train_y = self.bot_data['train_y']
		
		with open('Communicator/ChatBot/intents.json') as json_data:
			self.intents = json.load(json_data)


	def _load_model(self):
		tf.reset_default_graph()		
		self.net = tflearn.input_data(shape = [None, len(self.train_x[0])])
		self.net = tflearn.fully_connected(self.net, 8)
		self.net = tflearn.fully_connected(self.net, 8)
		self.net = tflearn.fully_connected(self.net, len(self.train_y[0]), activation = 'softmax')
		self.net = tflearn.regression(self.net)
		self.model = tflearn.DNN(self.net, tensorboard_dir = 'tflearn_logs')
		self.model.load('./Communicator/ChatBot/ChatBotModel/model.tflearn')


	def clean_up_sentences(self, sentence):
		sentence_words = nltk.word_tokenize(sentence)
		sentence_words = [self.stemmer.stem(word.lower()) for word in sentence_words]
		return sentence_words


	def bow(self, sentence, words, show_details = False):
		sentence_words = self.clean_up_sentences(sentence)
		bag = [0]*len(self.words)
		for s in sentence_words:
			for word_index, word in enumerate(self.words):
				if word == s:
					bag[word_index] = 1
					if show_details:
						print('found in bag: %s' % word)
		return np.array(bag)


	def classify(self, sentence):
		results = self.model.predict([self.bow(sentence, self.words)])[0]
		results = [[i, r] for i, r in enumerate(results) if r > self.ERROR_THRESHOLD]
		results.sort(key = lambda x: x[1], reverse = True)
		return_list = []
		for result in results:
			return_list.append((self.classes[result[0]], result[1]))
		return return_list


	def get_classification(self, sentence):
#		print('# LOG | ... BOT classifies message: %s' % sentence)
		results = self.classify(sentence)
		if results:
			while results:
				for i in self.intents['intents']:
					if i['tag'] == results[0][0]:
						return i['tag']
				results.pop(0)


	def response(self, sentence, user_id = '123'):
#		print('# LOG | ... BOT responds to message ...')
		results = self.classify(sentence)
		if results:
			while results:
				for i in self.intents['intents']:
					if i['tag'] == results[0][0]:
						return random.choice(i['responses'])
				results.pop(0)


	def get_drink_ready_message(self):
		return self.response('wwwwwwwwwww')
		

#============================================================

if __name__ == '__main__':

	sentence = 'I want a tequilasunrise'
	bot = Bot()
	print(bot.get_classification(sentence))
	print(bot.response(sentence))

