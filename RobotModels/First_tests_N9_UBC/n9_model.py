#!/usr/bin/env python 

import pickle 
import numpy as np 
import tensorflow as tf 
import edward as ed 
import uuid

from network_details import BayesianNeuralNetwork

#================================================================

class N9_Model(object):

	def __init__(self, name, date):
		self.name = name
		self.date = date
		self.id   = str(uuid.uuid4())
		self.experiments_remaining = 100

		self.parameters = []
		self.peak_areas = []

		self.bnn = BayesianNeuralNetwork()
		details  = pickle.load(open('model/training_set_specs.pkl', 'rb'))
		for att in ['mean_features', 'std_features', 'mean_targets', 'std_targets', 'max_targets', 'min_targets']:
			setattr(self.bnn.manager, att, details[att])

		self.bnn.construct_networks()
		self.sess  = ed.get_session()
		self.saver = tf.train.Saver()
		self.saver.restore(self.sess, 'model/bnn.ckpt')


	def run_experiment(self, parameters):

		if self.experiments_remaining <= 0:
			print('You already ran 100 experiments')
			return None

		if not len(parameters.shape) == 1:
			print('ERROR: parameters are expected to be an array, found shape', parameters.shape)
		if not len(parameters) == 6:
			print('ERROR: parameters needs to be a 6-dimensional vector, found %d entries' % len(parameters))

		predictions, uncertainties, _ = self.bnn.predict(parameters, n_post = 100, rescale = True)

		self.experiments_remaining -= 1

		self.parameters.append(parameters)
		self.peak_areas.append(predictions)
		print('chosen parameters:', parameters)
		print('achieved peak area: ', predictions)
		self.save_experiment()


	def save_experiment(self):

		parameters = np.array(self.parameters)
		peak_areas = np.array(self.peak_areas)

		file_name = 'n9_model_experiment_%s.pkl' % (self.id)
		var_dict = {'name': self.name, 'date': self.date, 'parameters': parameters, 'peak_areas': peak_areas}
		pickle.dump(var_dict, open(file_name, 'wb'))


	def print_summary(self, sort = False):

		if sort:
			indices = np.argsort(self.peak_areas)
			indices = indices[::-1]
		else:
			indices = np.arange(len(self.peak_areas))

		parameters = np.array(self.parameters)
		peak_areas = np.array(self.peak_areas)

		print_params = parameters[indices]
		print_values = peak_areas[indices]
		new_line = ''
		for index in range(len(self.peak_areas)):
			new_line += '# %d: \t %.2f \t %s\n' % (indices[index], print_values[index], str(print_params[index]))
		print(new_line)


#================================================================

if __name__ == '__main__':

	model = N9_Model('Flo', '02/06')
	params = np.random.uniform(low = 0., high = 1., size = 6)
	params = np.zeros(6) + 0.21
	print(params)
	model.run_experiment(params)
	model.save_experiment()

