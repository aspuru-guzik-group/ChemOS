#!/usr/bin/env python 

__author__ = 'Florian Hase'

#============================================================================

import os
import numpy as np 

from Utils.utils import ParserJSON, Printer

from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from smac.configspace import ConfigurationSpace
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC as SMAC_instance

#============================================================================

class SMAC(Printer):

	def __init__(self, config_file, work_dir):
		Printer.__init__(self, 'SMAC', color = 'grey')
		self.work_dir = work_dir
		self._parse_config_file(config_file)
		try:
			self.batch_size  = self.param_dict['general']['batches_per_round']
			self.num_batches = self.param_dict['general']['batch_size']
		except KeyError:
			self.num_batches = 1
		self._create_config_space()



	def rand_gens(self, var_type = 'float', size = 1):
		if var_type == 'float':
			return np.random.uniform(low = 0, high = 1, size = size)
		else:
			raise NotImplementedError



	def _parse_config_file(self, config_file):
		print(config_file)
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


	def _create_config_space(self):
		self.cs = ConfigurationSpace()
		for var_index, var_dict in enumerate(self.param_dict['variables']):
			variable = var_dict[self.var_names[var_index]]
			if variable['type'] == 'float':
				param = UniformFloatHyperparameter('x%d' % var_index, variable['low'], variable['high'])#, default = np.random.uniform(low = variable['low'], high = variable['high'], size = variable['size']))
			else:
				raise NotImplementedError()
			self.cs.add_hyperparameter(param)



	def _generate_uniform(self, num_samples = 10):
		self.container, self.sampled_params = {}, {}
		values = []
		for var_index, var_name in enumerate(self.var_names):
			sampled_values = self.rand_gens(var_type = self.param_dict['variables'][var_index][var_name]['type'], size = (self.param_dict['variables'][var_index][var_name]['size'], num_samples))
			values.extend(sampled_values)
			self.container[var_name] = sampled_values
		values = np.array(values)
		self.proposed = values.transpose()



	def _parse_observations(self, observations):
		all_params, all_losses = [], []
		for observation in observations:
			all_losses.append(observation['loss'])
			params = []
			for var_name in self.var_names:
				params.extend(observation[var_name]['samples'])
			all_params.append(params)
		return all_params, all_losses




	def _create_smac_instance(self):
		scenario = Scenario({'run_obj': 'quality',
							 'runcount-limit': 500,
							 'cs': self.cs,
							 'deterministic': 'true'})
		self.smac = SMAC_instance(scenario = scenario, rng = np.random.RandomState(np.random.randint(0, 10**5)))



	def _sample_parameter_sets(self, num_samples, observations):
		all_params, all_losses = self._parse_observations(observations)
		self._create_smac_instance()

		# get next parameter point
		challengers = self.smac.solver.choose_next(np.array(all_params), np.array(all_losses), np.amin(all_losses))
		self.proposed = []
		for index in range(self.num_batches * self.batch_size):
			self.proposed.append(challengers.challengers[index]._vector)



	def choose(self, num_samples = None, observations = None):
		current_dir = os.getcwd()
		os.chdir(self.work_dir)

		if not num_samples:
			num_samples = self.param_dict['general']['batches_per_round']
		if observations:
			self._print('proposing samples')
			self._sample_parameter_sets(num_samples, observations)
		else:
			self._print('choosing uniformly')
			self._generate_uniform(num_samples)

		os.chdir(current_dir)
		return self.proposed
