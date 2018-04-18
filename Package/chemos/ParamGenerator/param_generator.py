#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import copy
import numpy as np
from threading import Thread 

from DatabaseManager.database import Database
from Utils.utils import Printer

#========================================================================


#========================================================================

class ParamGenerator(Printer):

	BUSY             = {}
	DB_ATTRIBUTES    = {'exp_identifier': 'string',
					    'status': 'integer',
					    'parameters': 'pickle',
					    'sampling_parameter_value': 'integer'}
	OPTIMIZER        = None
	PARAM_STATUS     = {}
	TARGET_SPECS     = {}


	# add generated parameters as attributes to the parameter generator with a unique attribute name
	# then, have chemOS pick up the generated parameters and dump them in the respective database
	# make sure, to always properly destroy the attributes after they have been dumped



	def __init__(self, settings, verbose = True):
		Printer.__init__(self, 'PARAMETER GENERATOR')
		self.settings = settings
		self.verbose  = verbose

		self.settings['algorithm']['scratch_dir'] = self.settings['scratch_dir']

		# importing the wrappers here allows to use one optimimization algorithm
		# without having installed the others

		if self.settings['algorithm']['name'] == 'phoenics':
			from ParamGenerator.Phoenics.phoenics_wrapper import PhoenicsWrapper
			self.optimization_algorithm = PhoenicsWrapper(self.settings['algorithm'])
		elif self.settings['algorithm']['name'] == 'smac':
			from ParamGenerator.SMAC.smac_wrapper import SmacWrapper
			self.optimization_algorithm = SmacWrapper(self.settings['algorithm'])
		elif self.settings['algorithm']['name'] == 'spearmint':
			from ParamGenerator.Spearmint.spearmint_wrapper import SpearmintWrapper
			self.optimization_algorithm = SpearmintWrapper(self.settings['algorithm'])
		elif self.settings['algorithm']['name'] == 'random_search':
			from ParamGenerator.RandomSearch.random_search_wrapper import RandomsearchWrapper
			self.optimization_algorithm = RandomsearchWrapper(self.settings['algorithm'])
		else:
			raise NotImplementedError

		self.BUSY = {experiment['name']: False for experiment in self.settings['experiments']}
		self.number_proposed_parameters = {experiment['name']: 0 for experiment in self.settings['experiments']}

		self._create_database()
		self.number_proposed_recipes = {}



	def _create_database(self):
		db_settings   = self.settings['param_database']
		self.database = Database(db_settings['path'], self.DB_ATTRIBUTES, 
								 db_settings['database_type'], verbose = self.verbose)


	def _get_experiment(self, identifier):
		for experiment in self.settings['experiments']:
			if experiment['name'] == identifier:
				break
		return experiment


	def _get_random_parameters(self, identifier):
		# get experiment settings
		experiment = self._get_experiment(identifier)
		optimizer = self.optimization_algorithm.get_instance(identifier, experiment)
		normalized_parameter = optimizer.choose()

		parameter = self._rescale_parameters(normalized_parameter, identifier)
		# we need to rescale the
		return parameter



	def _get_sampling_parameter(self, identifier):
		if not identifier in self.number_proposed_parameters.keys():
			self.number_proposed_parameters[identifier] = 0
		return self.number_proposed_parameters[identifier] % self.settings['algorithm']['batch_size']




	def select_parameters(self, identifier):
		sampling_parameter = self._get_sampling_parameter(identifier)

		condition = {'exp_identifier': identifier, 'sampling_parameter_value': sampling_parameter, 'status': 0}
		target    = 'parameters'
		parameter = self.database.fetch(condition, target)
	
		# check if we got parameters from the database
		retrain = False
		if type(parameter).__name__ != 'ndarray':
			# we did not get a valid set of parameters, so we need to generate a random parameter set
			parameter = self._get_random_parameters(identifier)
			retrain   = True
		else:
			# update the status
			update = {'status': 1}
			self.database.update(condition, update)		

		wait = retrain and self.BUSY[identifier]
		
		parameter = np.squeeze(parameter)

		if not wait:
			self.number_proposed_parameters[identifier] += 1		
		return parameter, wait, retrain



	def remove_parameters(self, identifier):
		condition = {'exp_identifier': identifier}
		self.database.remove_all(condition)





	def _normalize_observations(self, observations, exp_identifier):
		# get experiment
		experiment = self._get_experiment(exp_identifier)
		var_names = [variable['name'] for variable in experiment['variables']]
		var_lows  = [variable['low'] for variable in experiment['variables']]
		var_highs = [variable['high'] for variable in experiment['variables']]

		rescaled_observations = []
		for observation in observations:
			rescaled_observation = copy.deepcopy(observation)
			for var_index, var_name in enumerate(var_names):
				value = rescaled_observation[var_name]['samples']
				# FIXME: for now, only linear rescaling
				rescaled_observation[var_name]['samples'] = (value - var_lows[var_index]) / (var_highs[var_index] - var_lows[var_index])
			rescaled_observations.append(rescaled_observation)
		return rescaled_observations



	def _rescale_parameters(self, normalized_parameters, exp_identifier):
		experiment = self._get_experiment(exp_identifier)
		var_names = [variable['name'] for variable in experiment['variables']]
		var_lows  = [variable['low'] for variable in experiment['variables']]
		var_highs = [variable['high'] for variable in experiment['variables']]
		var_sizes = [variable['size'] for variable in experiment['variables']]

		parameters = []
		for norm_param in normalized_parameters:
			start_index = 0
			param = []
			for var_index, var_name in enumerate(var_names):
				values   = norm_param[start_index : start_index + var_sizes[var_index]]
				# FIXME: for now, only linear rescaling
				rescaled = (var_highs[var_index] - var_lows[var_index]) * values + var_lows[var_index]
				param.extend(rescaled)
				start_index += var_sizes[var_index]
			parameters.append(np.copy(param))
		parameters = np.array(parameters)
		return parameters




	def kill_running_instances(self, exp_identifier):
		self._print('killing parameter generation for %s' % exp_identifier)
		# just need to stop the thread
		self.PARAM_STATUS[exp_identifier] = 'trash'



	def _parameter_generation(self):
		optimizer      = self._optimizer		
		exp_identifier = self._exp_identifier
		self._print('initializing learning procedure')

		# we need to rescale the parameters here!
		rescaled_observations = self._normalize_observations(self.TARGET_SPECS[exp_identifier], exp_identifier)
		normalized_parameters = optimizer.choose(observations = rescaled_observations)
		parameters = self._rescale_parameters(normalized_parameters, exp_identifier)

		if self.PARAM_STATUS[exp_identifier] == 'usable':

			# updating database
			self._print('updating parameter database')
			for parameter in parameters:
				print('\t', parameter, np.linalg.norm(parameter))
			condition   = {'exp_identifier': exp_identifier}
			new_entries = [{'exp_identifier': exp_identifier,
							'status': 0, 'parameters': parameters[index], 'sampling_parameter_value': index} for index in range(len(parameters))]
			self.database.replace(condition, new_entries)

		else:
			self._print('found only trash results')

		# reset
		del self.TARGET_SPECS[exp_identifier]
		del self.PARAM_STATUS[exp_identifier]
		self.BUSY[exp_identifier] = False
		del self._optimizer





	def generate_new_parameters(self, exp_identifier):
		for experiment in self.settings['experiments']:
			if experiment['name'] == exp_identifier:
				break

		# check if busy
		try:
			busy = self.BUSY[exp_identifier]
		except KeyError:
			busy = False
		if busy:
			return None
		self.BUSY[exp_identifier] = True

		self._print('starting parameter generation process for %s' % exp_identifier)
		self._print('getting optimizer instance')
		self._optimizer = self.optimization_algorithm.get_instance(exp_identifier, experiment)
		self._exp_identifier = exp_identifier
		self._print('submitting training process')

		# running the parameter generation locally 
		# FIXME: CHANGE CODE HERE TO IMPLEMENT TRAINING ON OTHER COMPUTING RESOURCES!
		generation_thread = Thread(target = self._parameter_generation)
		self.PARAM_STATUS[exp_identifier] = 'usable'
		generation_thread.start()
