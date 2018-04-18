#!/usr/bin/env pythoni

import numpy as np 
from scipy.optimize import minimize
from lazyme.string import color_print

import sys 
sys.path.append('./')
from .utils import ParserJSON, pickle_load, pickle_dump
from .bayesian_neural_network import ProbDist 

#============================================================================

TEMP = 2.0

def get_samples_out(args):
    proposals, batch_index, num_samples, penalty = args[0], args[1], args[2], args[3]
    reward  = lambda x: np.exp( - TEMP * penalty(x, batch_index))
    rewards = [reward(x) for x in proposals]
    sorted_reward_indices = np.argsort(rewards)
    return proposals[sorted_reward_indices[-num_samples:]]


#============================================================================

class Phoenics(object):

	def __init__(self, config_file = None):
		if config_file:
			self._parse_config_file(config_file)
		else:
			raise NotImplementedError
#		print(self.param_dict)
		try:
			self.num_batches = self.param_dict['general']['batch_size']
		except IndexError:
			self.num_batches = 1
		self.temp = TEMP
		self.temp_dec_factors = np.linspace(0.5, 0.9, self.num_batches)
		self.temp_dec_factors = np.zeros(self.num_batches) + 0.95
#		self.temp_factors = np.linspace(0.5, 2.0, self.num_batches)[::-1]**2
		self.temp_factors = np.zeros(self.num_batches) + TEMP


	def _print(self, message):
		message = 'PHOENICS ... %s ' % message
		color_print(message, color = 'grey')



	def _reset(self):
		self.container = {}
		self.sampled_params = {}


	def _parse_config_file(self, config_file):
		self.json_parser = ParserJSON(file_name = config_file)
		self.json_parser.parse()
		self.param_dict = self.json_parser.param_dict

		# now get the total number of variables 
		# and create a dictionary with the size of each variable
		self.total_size = 0
		self.var_sizes  = []
		self.var_names  = []
		for var_dict in self.param_dict['variables']:
			self.total_size += var_dict[list(var_dict)[0]]['size']
			self.var_sizes.append(int(var_dict[list(var_dict)[0]]['size']))
			self.var_names.append(list(var_dict)[0])


	def gen_arg(self, arg):
		return self.param_dict['general'][arg]


	def rand_gens(self, var_type = 'float', size = 1):
		if var_type == 'float':
			return np.random.uniform(low = 0, high = 1, size = size)
		else:
			raise NotImplementedError




	def _generate_uniform(self, num_samples = 10):
		self._reset()

		values = []
		for var_index, var_name in enumerate(self.var_names):
			sampled_values = self.rand_gens(var_type = self.param_dict['variables'][var_index][var_name]['type'], size = (self.param_dict['variables'][var_index][var_name]['size'], num_samples))
			values.extend(sampled_values)
			self.container[var_name] = sampled_values
		values = np.array(values)
		self.proposed_samples = values.transpose()


	def _get_mirrored_samples(self, sample, index_dict):
		index_dict_keys   = list(index_dict.keys())
		index_dict_values = list(index_dict.values())
		samples = []
		for index in range(2**len(index_dict)):
			sample_copy = np.copy(sample)
			for jndex in range(len(index_dict)):
#				sample_copy = np.copy(sample)
				if (index // 2**(jndex)) % 2 == 1:
					if index_dict_values[jndex] == 'lower':
						sample_copy[index_dict_keys[jndex]] = - sample_copy[index_dict_keys[jndex]]
					elif index_dict_values[jndex] == 'upper':
						sample_copy[index_dict_keys[jndex]] = 2 - sample_copy[index_dict_keys[jndex]]
			samples.append(sample_copy)
		if len(samples) == 0:
			samples.append(np.copy(sample))
		return samples


	def _separate_datasets(self, observations):
		losses  = []
		samples = []
		for observ_dict in observations:
	#		losses.append(observ_dict['loss'])
			sample = []
			for var_index, var_name in enumerate(self.var_names):
				sample.extend(observ_dict[var_name]['samples'])
			sample = np.array(sample)
			# now that we got the sample, we need to mirror it
			lower_indices = np.where(sample < 0.1)[0]
			upper_indices = np.where(sample > 0.9)[0]
			index_dict = {index: 'lower' for index in lower_indices}
			for index in upper_indices:
				index_dict[index] = 'upper'
			mirrored_samples = self._get_mirrored_samples(sample, index_dict)			
#			print('SAMPLES', len(mirrored_samples), mirrored_samples)
	#		quit()
			
			for sample in mirrored_samples:
				samples.append(np.array(sample))
				losses.append(observ_dict['loss'])


	#		samples.append(np.array(sample))
		losses  = np.array(losses)
		samples = np.array(samples)

#		losses = losses - np.amin(losses) + 1e-4
#		losses = np.log10(losses)

		if np.amin(losses) == np.amax(losses):
			losses -= np.amin(losses)
		else:
			losses = (losses - np.amin(losses)) / (np.amax(losses) - np.amin(losses))
		losses = np.sqrt(losses)

		return samples, losses





	def _sample_posterior(self, num_samples = 30):
		self._print('compiling penalties')
		self.generator.build_penalty()

		uniform_samples = np.random.uniform(low = 0., high = 1., size = ( 5 * self.total_size * num_samples, self.total_size))
		# extend proposals by perturbations
		self._print('generating perturbations')
#		indices = np.random.randint(low = 0, high = len(self.model_set), size = (num_samples))
		min_loss_index = np.argmin(self.losses)
		selected_samples  = self.model_set[min_loss_index]

		perturbed_samples_3 = selected_samples + np.random.normal(0., 0.5, size = ( 5 * self.total_size * num_samples, self.total_size))
		perturbed_samples_0 = selected_samples + np.random.normal(0., 0.1, size = ( 5 * self.total_size * num_samples, self.total_size))
		perturbed_samples_1 = selected_samples + np.random.normal(0., 0.05, size = (5 * self.total_size * num_samples, self.total_size))
		perturbed_samples_2 = selected_samples + np.random.normal(0., 0.01, size = (5 * self.total_size * num_samples, self.total_size))

		proposals = [sample.copy() for sample in uniform_samples]
		for sample in perturbed_samples_0:
			sample = np.abs(sample)
			sample = 1 - np.abs(1 - sample)
			if np.all(sample > 0.) and np.all(sample < 1.):
				proposals.append(sample.copy())

		for sample in perturbed_samples_1:
			sample = np.abs(sample)
			sample = 1 - np.abs(1 - sample)
			if np.all(sample > 0.) and np.all(sample < 1.):
				proposals.append(sample.copy())

		for sample in perturbed_samples_2:
			sample = np.abs(sample)
			sample = 1 - np.abs(1 - sample)
			if np.all(sample > 0.) and np.all(sample < 1.):
				proposals.append(sample.copy())

		for sample in perturbed_samples_3:
			sample = np.abs(sample)
			sample = 1 - np.abs(1 - sample)
			if np.all(sample > 0.) and np.all(sample < 1.):
				proposals.append(sample.copy())

		proposals = np.array(proposals)

		del uniform_samples
		del perturbed_samples_0
		del perturbed_samples_1
		del perturbed_samples_2
		del perturbed_samples_3

		self._print('getting general penalty properties')
		all_probs = np.array([self.generator.get_probs(sample.copy()) for sample in proposals])
		denoms    = np.array([np.sum(probs.copy()) + 1. for probs in all_probs])
		dot_prods = np.array([self.generator.get_dot_prods(probs.copy()) for probs in all_probs])

		del all_probs

		self._print('collecting samples')
		samples = []
		for batch_index in range(self.num_batches):
			penalties = (dot_prods + self.generator.lambda_values[batch_index]) / denoms
			rewards   = np.exp( - self.temp * penalties)
			sorted_reward_indices = np.argsort(rewards)

			to_optimize = proposals[sorted_reward_indices[-num_samples:]].copy()

#			samples.append(proposals[sorted_reward_indices[-num_samples:]])
			optimized = []
			for sample in to_optimize:
				if np.random.uniform() < 0.5:
					optimized.append(sample)
					continue				

				res = minimize(self.generator.penalties[batch_index], sample, method = 'L-BFGS-B', options = {'maxiter': 10})
				if np.any(res.x < 0) or np.any(res.x > 1):
					optimized.append(sample)
				else:
					optimized.append(res.x)

			samples.append(optimized)

		del proposals
		del denoms
		del dot_prods
	
		self.proposed_samples = np.array(samples)





#	def _sample_posterior_slow(self, num_samples = 100):
#		self._print('compiling penalties')
#		self.generator.build_penalty()
#
#		def get_samples(args):
#			proposals, batch_index, num_samples = args[0], args[1], args[2]
#			reward  = lambda x: np.exp( - self.temp * self.generator.penalty(x, batch_index))
#			rewards = [reward(x) for x in proposals]
#			sorted_reward_indices = np.argsort(rewards)
#			return proposals[sorted_reward_indices[-num_samples:]] 
#
#		all_args = []
#		for batch_index in range(self.num_batches):
#			all_args.append([np.random.uniform(low = 0, high = 1., size = (5 * 10**2 * num_samples, self.total_size)), batch_index, num_samples, self.generator.penalty])

#		self._print('getting samples')
#		samples = []
#		for batch_index in range(self.num_batches):
#			samples.append(get_samples(all_args[batch_index]))
#		self.proposed_samples = np.array(samples)





#	def _sample_posterior_sequential(self, num_samples = 100):
#
#		# first we get the penalty of the generator
#		self._print('compiling penalties')
#		self.generator.build_penalty()
#
#		self._print('generating samples')
#		samples = []
#		for batch_index in range(self.num_batches):
#			self._print('batch index %d' % batch_index)
#
#			batch_samples = []
#			penalty = lambda x: self.generator.penalty(x, batch_index)
#			reward  = lambda x: np.exp( - self.temp * self.generator.penalty(x, batch_index))
#
#			index = 0
#			current_temp = self.temp_factors[batch_index]
#
#			x = np.random.uniform(low = 0., high = 1., size = (5 * 10**3 * num_samples, self.total_size))
#			y = np.random.uniform(low = 0., high = 1., size = (1000 * num_samples)) * np.amax([1., np.exp(current_temp)])
#
#			rewards = [reward(sample) for sample in x]
#
#			# sort rewards
#			sorted_reward_indices = np.argsort(rewards)
#
#			batch_samples = x[sorted_reward_indices[-num_samples:]]
#
#			samples.append(np.array(batch_samples))
#		self.proposed_samples = np.array(samples)		# of shape (# batches, # samples, sample dim)
#		quit()




	def _clean_samples(self, num_samples):

		# build penalties
#		self.generator.build_penalty()
		
		all_samples = []
		self.clean_samples = []
		for batch_index in range(self.num_batches):

			batch_samples = []
			current_temp = self.temp_factors[batch_index]
			while len(batch_samples) < num_samples:
				min_probs = []
				for sample in self.proposed_samples[batch_index]:
					if len(all_samples) > 0:
						min_distance = np.amin([np.linalg.norm(sample - x) for x in all_samples])
					else:
						min_distance = self.generator.characteristic_distance

						
					min_prob = np.amin( np.exp( - current_temp * self.generator.penalty(sample, batch_index)) )
					min_prob *= np.amin([1., np.exp(10 * (min_distance - self.generator.characteristic_distance))])
					min_probs.append(min_prob)

				sorted_indices = np.argsort(min_probs)[::-1]
				samples   = self.proposed_samples[batch_index][sorted_indices].copy()
				min_probs = np.array(min_probs)[sorted_indices]

				sample_index = 0
				batch_samples = []
				while len(batch_samples) < num_samples and sample_index < len(min_probs):
#					if np.random.uniform() * np.amax([1., np.exp(current_temp)]) < min_probs[sample_index]:
					batch_samples.append(samples[sample_index])
					all_samples.append(samples[sample_index])
					sample_index += 1

				if len(batch_samples) < num_samples:
					current_temp *= self.temp_dec_factors[batch_index]#0.5
#					self._print('decreased temperature to ', current_temp, batch_index)

			self.clean_samples.append(np.array(batch_samples))
		self.clean_samples = np.array(self.clean_samples)

		cs_shape = self.clean_samples.shape
		self.clean_samples = np.reshape(self.clean_samples, (cs_shape[0] * cs_shape[1], cs_shape[2]))

		del self.proposed_samples


	def _sample_parameter_sets(self, num_samples, observ_dict):
#		self._print('setting up generator')

		model_set, losses = self._separate_datasets(observ_dict)
		self.model_set = model_set
		self.losses = losses
		self.generator = ProbDist(model_set, losses, self.num_batches)

		# create model
		self._print('creating model')
		self.generator.create_model()

		# train model
		self._print('sampling models')
		self.generator.sample()

		# propose sample
		self._print('proposing samples')
		self._sample_posterior()

		# clean samples
		self._print('cleaning samples')
		self._clean_samples(num_samples)

		del self.generator.mus
		del self.generator.sds
		del self.generator.penalty
		del self.generator.prob
		del self.generator.probs


	def choose(self, num_samples = None, observations = None):
		if not num_samples:
			num_samples = self.param_dict['general']['batches_per_round']
		if observations:
			self._print('proposing samples')
			self._sample_parameter_sets(num_samples, observations)
		else:
			self._generate_uniform(num_samples)
			self.clean_samples = self.proposed_samples

		return self.clean_samples


#============================================================================

if __name__ == '__main__':

	chooser = Chooser('config.txt')
	sets = chooser.choose(num_samples = 10, observed = pickle_load('evaluated_sets.pkl'))
