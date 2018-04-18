#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import copy
import numpy as np 
import pickle

from DatabaseManager.database import Database
from Utils.utils import Printer

#========================================================================

class FeedbackHandler(Printer):

	DB_ATTRIBUTES = {'status': 'string',
					 'job_id': 'string', 
					 'repetition': 'integer',
					 'work_dir': 'string',
					 'exp_identifier': 'string',
					 'parameters': 'pickle',
					 'loss': 'float',
					 'author': 'pickle'}



	def __init__(self, settings, verbose = True):
		Printer.__init__(self, 'FEEDBACK HANDLER', color = 'yellow')
		self.settings = settings
		self.verbose  = verbose

		self._create_database()


	def _create_database(self):
		db_settings   = self.settings['feedback_database']
		self.database = Database(db_settings['path'], self.DB_ATTRIBUTES,
								 db_settings['database_type'], verbose = self.verbose)


	def process_feedback(self, feedback_dict):
		feedback_dict['status'] = 'new'
		self.database.add(feedback_dict)


	def remove_feedback(self, identifier):
		self._print('removing feedback for %s' % identifier)
		condition = {'exp_identifier': identifier}
		self.database.remove_all(condition)


	def get_new_feedback(self):
		condition         = {'status': 'new'}
		new_feedback_list = self.database.fetch_all(condition)
		# separate the new feedbacks by name
		new_feedbacks = {}
		for feedback in new_feedback_list:
			if feedback['exp_identifier'] in new_feedbacks.keys():
				new_feedbacks[feedback['exp_identifier']].append(feedback)
			else:
				new_feedbacks[feedback['exp_identifier']] = [feedback]
		return new_feedbacks



	def _construct_observation_dict(self, exp_identifier, parameters, feedback, work_dirs):
		# get experiment information
		for experiment in self.settings['experiments']:
			if experiment['name'] == exp_identifier:
				break
		# collect observations
		observations = []
		for index, parameter in enumerate(parameters):
			start_index = 0
			new_obs = {'loss': feedback[index], 'work_dir': work_dirs[index]}
			for variable in experiment['variables']:
				if len(parameter.shape) == 1:
					new_obs[variable['name']] = {'samples': parameter[start_index : start_index + variable['size']]}
				elif len(parameter.shape) == 2:
					new_obs[variable['name']] = {'samples': parameter[0, start_index : start_index + variable['size']]}
				start_index += variable['size']
			observations.append(copy.deepcopy(new_obs))
		return observations
				


	def get_observations(self, exp_identifier):
		condition  = {'exp_identifier': exp_identifier}
		entry_list = self.database.fetch_all(condition)
		parameters, feedbacks, work_dirs = [], [], []
		for entry in entry_list:
			work_dirs.append(entry['work_dir'])
			parameters.append(entry['parameters'])
			feedbacks.append(entry['loss'])
		parameters = np.array(parameters)
		feedbacks  = np.array(feedbacks)

		# observations: list of dictionaries with 'loss' and var_name being the key for another dict
		observations = self._construct_observation_dict(exp_identifier, parameters, feedbacks, work_dirs)

		return observations



	def set_all_to_used(self, exp_identifier):
		condition = {'exp_identifier': exp_identifier, 'status': 'new'}
		update    = {'status': 'used'}  
		self.database.update(condition, update)



#==================================================================

