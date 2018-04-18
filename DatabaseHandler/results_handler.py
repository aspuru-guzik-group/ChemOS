#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import copy
import numpy as np 
import pickle

from DatabaseManager.database import Database
from Utilities.misc import Printer

#========================================================================

class ResultsHandler(Printer):

	DB_ATTRIBUTES = {'status': 'string',
					 'job_id': 'string', 
					 'repetition': 'integer',
					 'work_dir': 'string',
					 'exp_identifier': 'string',
					 'parameters': 'pickle',
					 'objectives': 'pickle',
					 'author': 'pickle'}
	PROCESSED_JOBS = []


	def __init__(self, settings, verbose = True):
		Printer.__init__(self, 'RESULTS HANDLER', color = 'yellow')
		self.settings = settings
		self.verbose  = verbose 

		self._create_database()


	def _create_database(self):
		db_settings   = self.settings['results_database']
		self.database = Database(db_settings['path'], self.DB_ATTRIBUTES,
								 db_settings['database_type'], verbose = self.verbose)



	def process_results(self, results_dict):
		results_dict['status'] = 'new'
		self.database.add(results_dict)


	def remove_results(self, identifier):
		self._print('removing feedback for %s' % identifier)
		condition = {'exp_identifier': identifier}
		self.database.remove_all(condition)


	def get_new_results(self):
		condition        = {'status': 'new'}
		new_results_list = self.database.fetch_all(condition)
		
		# check, if: 
		#  - for a given experiment
		#  - and a given job_id
		#  --> all repetitions are executed
		new_results = {}
		for result in new_results_list:
			exp_identifier = result['exp_identifier']
			job_id         = result['job_id']
			if exp_identifier in new_results.keys():
				if job_id in new_results[exp_identifier]:
					new_results[exp_identifier][job_id].append(result)
				else:
					new_results[exp_identifier][job_id] = [result]
			else:
				new_results[exp_identifier] = {job_id: [result]}


		# get those jobs, for which we have all the results
		completed_jobs = []
		for exp_identifier in new_results.keys():
			# get experiment
			for experiment in self.settings['experiments']:
				if experiment['name'] == exp_identifier:
					break
			num_repetitions = experiment['repetitions']
			for job_id in new_results[exp_identifier]:
				if len(new_results[exp_identifier][job_id]) == num_repetitions:
					completed_jobs.append(job_id)

		return completed_jobs




		# separate the new feedbacks by name and by repetition
#		new_results = {}
#		condition         = {'status': 'new'}
#		new_result_list = self.database.fetch_all(condition)
		# separate the new feedbacks by name
#		new_results = {}
#		for result in new_result_list:
#			if result['exp_identifier'] in new_results.keys():
#				new_results[result['exp_identifier']].append(result)
#			else:
#				new_results[result['exp_identifier']] = [result]
#		return new_results



	def analyze_new_results(self, job_id):
		# get experiments with the defined job_id
		condition = {'job_id': job_id}
		results   = self.database.fetch_all(condition)

		# copy information to the processed dictionary
		processed = {}
		for att in ['job_id', 'work_dir', 'exp_identifier', 'parameters', 'author']:
			processed[att] = copy.deepcopy(results[0][att])
		processed['loss'] = {}


		# perform operations on results
		exp_identifier = results[0]['exp_identifier']
		for experiment in self.settings['experiments']:
			if experiment['name'] == exp_identifier:
				break

		for objective in experiment['objectives']:
			name      = objective['name']
			operation = objective['operation']

			# get all results
#			print('RESULT', results)
			values = np.array([result['objectives'][name] for result in results])
			if operation == 'average':
				value = np.mean(values)
			elif operation == 'std_rel':
				value = np.std(values) / np.mean(values)
			else:
				raise NotImplementedError()
			processed['loss']['%s_%s' % (name, operation)] = value

		setattr(self, 'info_dict_%s' % job_id, copy.deepcopy(processed))
		self.PROCESSED_JOBS.append(job_id)


	def set_all_to_used(self, job_id):
		condition = {'job_id': job_id, 'status': 'new'}
		update    = {'status': 'used'}
		self.database.update(condition, update)

#	def set_all_to_used(self, exp_identifier):
#		condition = {'exp_identifier': exp_identifier, 'status': 'new'}
#		update    = {'status': 'used'}  
#		self.database.update(condition, update)

#========================================================================
