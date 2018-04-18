
ERROR_THRESHOLD = 0.25
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# things we need
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


# restore previous data
import pickle 
data    = pickle.load(open('training_data', 'rb'))
words   = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import chatbot
import json 
with open('intents.json') as json_data:
	intents = json.load(json_data)

# load saved model
import tensorflow as tf 
import tflearn
import numpy as np
tf.reset_default_graph()
net = tflearn.input_data(shape = [None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation = 'softmax')
net = tflearn.regression(net)
# define model and setup
model = tflearn.DNN(net, tensorboard_dir = 'tflearn_logs')
model.load('./ChatBotModel/model.tflearn') 

# utilities
def clean_up_sentences(sentence):
	sentence_words = nltk.word_tokenize(sentence)
	sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
	return sentence_words

def bow(sentence, words, show_details = False):
	sentence_words = clean_up_sentences(sentence)
	bag = [0]*len(words)
	for s in sentence_words:
		for i, w in enumerate(words):
			if w == s:
				bag[i] = 1
				if show_details:
					print('found in bag: %s' % w)
	return np.array(bag)


def classify(sentence):
	results = model.predict([bow(sentence, words)])[0]
	results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
	results.sort(key = lambda x: x[1], reverse = True)
	return_list = []
	for r in results:
		return_list.append((classes[r[0]], r[1]))
	return return_list


def response(sentence, user_id = '123'):
	results = classify(sentence)
	if results:
		while results:
			for i in intents['intents']:
				if i['tag'] == results[0][0]:
					print i
					return [i['tag'], random.choice(i['responses'])]
			results.pop(0)


import sys
#print(classes)
#print(classify(sys.argv[1]))
print '>>', response(sys.argv[1])



