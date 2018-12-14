#!/usr/bin/env python 

__author__ = 'Florian Hase'

#============================================================================

import numpy as np 

from Utilities.misc import ParserJSON, Printer

#============================================================================


class RandomSearcher(Printer):
	
	def __init__(self, config_file):
		super(RandomSearcher, self).__init__('RandomSearcher', color = 'grey')
		self._parse_config_file(config_file)

		try:
			self.batch_size  = self.param_dict['general']['batches_per_round']
			self.num_batches = self.param_dict['general']['batch_size']
		except KeyError:
			self.num_batches = 1
		
		# we might also have to set the random seed
		try:
			self.random_seed = self.param_dict['general']['random_seed']
		except KeyError:
			# no random seed provided
			pass




	def rand_gens(self, var_type = 'float', size = 1):
	
		# if provided, set the random seed
		if hasattr(self, 'random_seed'):
			np.random.seed(self.random_seed)

		if var_type == 'float':
			samples = np.random.uniform(low = 0, high = 1, size = size)
		else:
			raise NotImplementedError

		# if provided, update the random seed
		if hasattr(self, 'random_seed'):
			new_random_seed  = np.random.randint(low = 0, high = 2**30)
			self.random_seed = new_random_seed

		return samples


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


	def _generate_uniform(self, num_samples = 10):
		self.container, self.sampled_params = {}, {}
		values = []
		for var_index, var_name in enumerate(self.var_names):
			sampled_values = self.rand_gens(var_type = self.param_dict['variables'][var_index][var_name]['type'], size = (self.param_dict['variables'][var_index][var_name]['size'], num_samples))
			values.extend(sampled_values)
			self.container[var_name] = sampled_values
		values = np.array(values)
		self.proposed = values.transpose()




	def choose(self, num_samples = None, observations = None):
		if not num_samples:
			num_samples = self.param_dict['general']['batches_per_round']
		
		# we don't care about observations - this is random search!
		self._print('choosing uniformly')
		self._generate_uniform(num_samples)

		return self.proposed

