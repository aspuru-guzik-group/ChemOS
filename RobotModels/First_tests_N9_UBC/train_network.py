#!/usr/bin/env python 

import pickle 
import numpy as np 

import tensorflow as tf 
import edward as ed 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#==============================================================

class DataManager(object):

	def __init__(self, file_name):

		data = pickle.load(open(file_name, 'rb'))
		self.params   = data['params']
		self.values  = data['values']

		self.features = data['params']
		self.targets  = data['values']
		self.targets  = np.reshape(self.targets, (len(self.targets), 1))
		self._split_dataset()


	def _split_dataset(self):
		np.random.seed(100691)

		self.mean_features = np.mean(self.features, axis = 0)
		self.std_features  = np.std(self.features, axis = 0)
		self.features      = (self.features - self.mean_features) / self.std_features

		self.mean_targets  = np.mean(self.targets)
		self.std_targets   = np.std(self.targets)
#		self.targets       = (self.targets - self.mean_targets) / self.std_targets

		self.max_targets   = np.amax(self.targets)
		self.min_targets   = np.amin(self.targets)
		self.targets       = (self.targets - self.min_targets) / (self.max_targets - self.min_targets)

		pos = np.arange(len(self.features))
		np.random.shuffle(pos)

		num_points = len(self.features)
		num_train  = int(0.8 * num_points)
		num_valid  = int(0.2 * num_points)

		self.train_features = self.features[pos[:num_train]]
		self.train_targets  = self.targets[pos[:num_train]]
		self.valid_features = self.features[pos[num_train:]]
		self.valid_targets = self.targets[pos[num_train:]]

#==============================================================

if __name__ == '__main__':

	import time
	from network_details import BayesianNeuralNetwork

	network = BayesianNeuralNetwork()
	network.manager = DataManager('RawData/n9_data.pkl')
	network.train_features = network.manager.train_features
	network.train_targets = network.manager.train_targets
	network.valid_features = network.manager.valid_features
	network.valid_targets = network.manager.valid_targets
	network.construct_networks()

	all_train_losses, all_valid_losses = [], []

	# store some details in the pickle file
	data = {'mean_features': network.manager.mean_features, 'std_features': network.manager.std_features,
			'mean_targets': network.manager.mean_targets, 'std_targets': network.manager.std_targets,
			'min_targets': network.manager.min_targets, 'max_targets': network.manager.max_targets}
	pickle.dump(data, open('model/training_set_specs.pkl', 'wb'))

	stride = 1
	saver = tf.train.Saver()

	train_loss_list = [10**12]
	valid_loss_list = [10**12]

	for epoch in range(10**6):

		start = time.time()
		network.train(10**4, epoch)

		newline = 'time spent training: %.2f\n' % (time.time() - start)
		print(newline[:-1])

		start = time.time()	
		train_pred, valid_pred, post_vars = network._predict()
		newline = 'time spent predicting: %.2f\n' % (time.time() - start)
		print(newline[:-1])

		# get losses 
		train_losses, valid_losses = [], []
		for i in range(len(train_pred)):
			train_loss = np.mean(np.abs(train_pred[i] - network.train_targets[::stride]))
			train_losses.append(train_loss)
			valid_loss = np.mean(np.abs(valid_pred[i] - network.valid_targets[::stride]))
			valid_losses.append(valid_loss)
		all_train_losses.append(np.array(train_losses))
		all_valid_losses.append(np.array(valid_losses))

		train_loss_array = np.array(all_train_losses)
		valid_loss_array = np.array(all_valid_losses)

		last_train_loss = np.mean(train_loss_array, axis = 1)[-1]
		last_valid_loss = np.mean(valid_loss_array, axis = 1)[-1]

		print_train_loss = np.mean(np.abs( np.mean(train_pred, axis = 0) - network.train_targets))
		print_valid_loss = np.mean(np.abs( np.mean(valid_pred, axis = 0) - network.valid_targets))


		if (print_train_loss + print_valid_loss) < (train_loss_list[-1] + valid_loss_list[-1]):
			sess = ed.get_session()
			save_path = saver.save(sess, './model/bnn.ckpt')
			print('... saved model')
		else:
			sess = ed.get_session()
			save_path = saver.save(sess, './model/tmp_bnn.ckpt')

		train_loss_list.append(print_train_loss)
		valid_loss_list.append(print_valid_loss)

		newline = '%d\t%.5e\t%.5e\n' % (epoch, print_train_loss, print_valid_loss)

		print(newline)


