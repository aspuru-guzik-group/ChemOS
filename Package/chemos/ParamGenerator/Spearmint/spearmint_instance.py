#!/usr/bin/env python 

__author__ = 'Florian Hase'

#============================================================================

import os
import sys
import time
import uuid
import shutil 
import importlib
import subprocess
import numpy as np 

from Utils.utils import Printer, ParserJSON

#============================================================================

home = os.getcwd()
sys.path.append('%s/ParamGenerator/Spearmint/' % home)
sys.path.append('%s/ParamGenerator/Spearmint/spearmint' % home)
for directory in ['kernels', 'models', 'sampling', 'schedulers', 'transformations', 'utils']:
	sys.path.append('%s/ParamGenerator/Spearmint/spearmint/%s' % (home, directory))
sys.path.append('%s/ParamGenerator/Spearmint/spearmint/utils/database' % home)

from main import get_options, get_suggestion, parse_db_address, save_job
from spearmint.utils.database.mongodb import MongoDB
from spearmint.resources.resource import parse_resources_from_config

#============================================================================

class Spearmint(Printer):

	def __init__(self, config_file, work_dir):
		Printer.__init__(self, 'Spearmint', color = 'grey')
		self.work_dir = work_dir
		print(config_file)
		self._parse_config_file(config_file)
		try:
			self.batch_size  = self.param_dict['resources']['my-machine']['max-concurrent']
#			self.num_batches = self.param_dict['general']['batches_per_round']
		except KeyError:
#			self.num_batches = 1
			self.batch_size  = 1
		self.all_params, self.all_losses = [], []



	def rand_gens(self, var_type = 'float', size = 1):
		if var_type == 'float':
			return np.random.uniform(low = 0, high = 1, size = size)
		else:
			raise NotImplementedError


	def _parse_config_file(self, config_file):
		self.json_parser = ParserJSON(file_name = config_file)
		self.json_parser.parse()
		self.param_dict = self.json_parser.param_dict

		# now get the total number of variables 
		# and create a dictionary with the size of each variable
		self.total_size = 0
		self.var_sizes  = []
		self.var_names  = []
		for var_name, var_dict in self.param_dict['variables'].items():
			self.total_size += var_dict['size']
			self.var_sizes.append(int(var_dict['size']))
			self.var_names.append(var_name)

#			self.total_size += var_dict[list(var_dict)[0]]['size']
#			self.var_sizes.append(int(var_dict[list(var_dict)[0]]['size']))
#			self.var_names.append(list(var_dict)[0])
#


	def _generate_uniform(self, num_samples = 10):
		self.container, self.sampled_params = {}, {}
		values = []
		for var_index, var_name in enumerate(self.var_names):
			sampled_values = self.rand_gens(var_type = self.param_dict['variables'][var_name]['type'], size = (self.param_dict['variables'][var_name]['size'], num_samples))
			values.extend(sampled_values)
			self.container[var_name] = sampled_values
		values = np.array(values)
		self.proposed = values.transpose()


	def _parse_observations(self, observations):
		all_params, all_losses = [], []
		for observation in observations:
			params = []
			for var_name in self.var_names:
				params.extend(observation[var_name]['samples'])
			if len(self.all_params) > 0:
				if np.amin([np.linalg.norm(params - old_param) for old_param in self.all_params]) > 1e-6:
					all_losses.append(observation['loss'])
					all_params.append(params)
			else:
				all_losses.append(observation['loss'])
				all_params.append(params)
		for index, element in enumerate(all_params):
			self.all_params.append(element)
			self.all_losses.append(all_losses[index])
		return all_params, all_losses


	def _create_mongo_instance(self):
		self.db_path = '%s/db_%s/' % (self.work_dir, self.param_dict['experiment-name'])
		print(self.db_path)
		try:
			shutil.rmtree(self.db_path)
		except:
			pass
		os.mkdir(self.db_path)
		subprocess.call('mongod --fork --logpath %s/mongodb.log --dbpath %s' % (self.db_path, self.db_path), shell = True)


	def _create_spearmint_parameters(self):
		self._create_mongo_instance()
		self.options, self.exp_dir = get_options(self.work_dir)
		self.resources        = parse_resources_from_config(self.options)
		self.chooser_module   = importlib.import_module('spearmint.choosers.' + self.options['chooser'])
		self.chooser          = self.chooser_module.init(self.options)
		self.experiment_name  = self.options.get('experiment-name', 'unnamed_experiment')

		self.db_address = self.options['database']['address']
		self.db = MongoDB(database_address = self.db_address)



	def _sample_parameter_sets(self, num_samples, observations):
		all_params, all_losses = self._parse_observations(observations)
		self._create_spearmint_parameters()

		# dump all observations in database
		for index, param in enumerate(all_params):
			print('PARAM', param, all_losses[index])
			params = {}
			start_index = 0
			for var_index, var_name in enumerate(self.var_names):
				var_dict = self.param_dict['variables'][var_name]
				params[var_name] = {'type': var_dict['type'], 'values': np.array(param[start_index : start_index + var_dict['size']])}
				start_index += var_dict['size']
			job = {'id': index + 1, 'expt_dir': self.work_dir, 'tasks': ['main'], 'resource': 'my-machine', 'main-file': 'main_file.py',
				   'language': 'PYTHON', 'status': 'new', 'submit time': time.time(), 'start time': time.time(), 'end time': None,
				   'params': params}
			time.sleep(0.1)
			job['values'] = {'main': all_losses[index]}
			job['status'] = 'complete'
			job['end time'] = time.time()

#			for key, value in job.items():
#				print(key, value)

			self.db.save(job, self.experiment_name, 'jobs', {'id': job['id']})



		self.proposed = []
		for resource_name, resource in self.resources.items():
			print('RUNNING SPEARMINT')
			suggested_job = get_suggestion(self.chooser, resource.tasks, self.db, self.exp_dir, self.options, resource_name)
			print('DONE')
			vector = []
			for var_name in self.var_names:
				vector.extend(suggested_job['params'][var_name]['values'])
			vector = np.array(vector)
			for index in range(num_samples):
				self.proposed.append(vector)

		print('PROPOSED', self.proposed)
		subprocess.call('mongod --shutdown --logpath %s/mongodb.log --dbpath %s' % (self.db_path, self.db_path), shell = True)	



	def choose(self, num_samples = None, observations = None):
		current_dir = os.getcwd()
		os.chdir(self.work_dir)

		if not num_samples:
			num_samples = self.batch_size

		if observations:
			self._print('proposing samples')
			self._sample_parameter_sets(num_samples, observations)
		else:
			self._print('choosing uniformly')
			self._generate_uniform(1)

		os.chdir(current_dir)		

#		print('SHAPE', self.proposed.shape)
		return self.proposed
