#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import os, copy
import pickle
import shutil
import subprocess

from DatabaseManager.database import Database
from Utilities.file_logger import FileLogger
from Utilities.misc import Printer

#========================================================================

class BotManager(Printer):

	DB_ATTRIBUTES = {'status': 'integer',
					 'name': 'string', 						# serves as id (i.e. is unique)
					 'parameters': 'pickle',				# list of parameters the bot can process
					 'possible_experiments': 'pickle', 		# list of possible experiments
					 'communication': 'dictionary'} 			# manages communication with bot
	QUEUED_JOBS    = []
	PROCESSED_JOBS = []
	FILE_LOGGERS   = []
	RUNNING_EXPS   = []
	ROBOT_STATUS   = {}
					 

	def __init__(self, settings, verbose = True):
		Printer.__init__(self, 'BOT MANAGER', color = 'red')
		self.settings = settings
		self.verbose = verbose
		self._create_database()

		# add all bots found in settings
		self._add_all_bots()


	def _create_database(self):
		db_settings   = self.settings['bot_database']
		self.database = Database(db_settings['path'], self.DB_ATTRIBUTES, 
								 db_settings['database_type'], verbose = self.verbose)


	def _add_all_bots(self):
		for bot_dict in self.settings['bots']:
			bot_dict['status'] = 0
			self.add_bot(bot_dict)


	def add_bot(self, bot_dict):
		# we need to find out which experiments the bot can run
		bot_dict['possible_experiments'] = []
		for experiment in self.settings['experiments']:
			possible = True
			for variable in experiment['variables']:
				possible = possible and variable['name'] in bot_dict['parameters']
			if possible:
				bot_dict['possible_experiments'].append(experiment['name'])
		self.database.add(bot_dict)
					 

	def get_available(self, experiment_name):
		condition      = {'status': 0}
		available_bots = self.database.fetch_all(condition)

		# get bots matching to experiments
		for bot in available_bots:
			if experiment_name in bot['possible_experiments']:
				break
		else:
			# did not find bot
			return None
		return bot



	def _relabel(self, bot_name, status):
		condition = {'name': bot_name}
		update    = {'status': status}
		return self.database.update(condition, update)


	def label_busy(self, bot_info):
		if isinstance(bot_info, dict):	
			return self._relabel(bot_info['name'], 1)
		elif isinstance(bot_info, str):
			return self._relabel(bot_info, 1)


	def label_available(self, bot_info):
		if isinstance(bot_info, dict):
			return self._relabel(bot_info['name'], 0)
		elif isinstance(bot_info, str):
			return self._relabel(bot_info, 0)



	def process_evaluated_params(self, file_name):
		# file logger finds the file it is looking for 
		# and triggers this method 
		# which collects and processes all information
		self._print('found processed job in %s' % file_name)

		# see if this is a loadable and comprehensible pickle file
		try:
			data     = pickle.load(open(file_name, 'rb'))
			job_id   = data['job_id']
			bot_name = data['bot_name']
			exp_name = data['exp_identifier']
		except AttributeError:
			self._print('could not process file %s' % file_name)
			return None

		# check if we want to use the file
		if self.ROBOT_STATUS[exp_name] == 'usable':

			# getting the loss
			for experiment in self.settings['experiments']:
				if exp_name == experiment['name']:
#					loss_name   = experiment['loss_name']
#					loss_type   = experiment['loss_type']
					repetitions = experiment['repetitions']
					break

			data['objectives'] = {}
			for objective in experiment['objectives']:
				try:
					data['%s_raw' % objective['name']] = copy.deepcopy(data[objective['name']])
				except: 
					self._print('could not process file %s' % file_name)
					return None

				if objective['type'] == 'minimum':
					data['objectives'][objective['name']] = copy.deepcopy(data[objective['name']])
				elif objective['type'] == 'maximum':
					data['objectives'][objective['name']] = - copy.deepcopy(data[objective['name']])
				else:
					raise NotImplementedError


			# store data dict as attribute and append to processed jobs
			setattr(self, 'info_dict_%s_%d' % (data['job_id'], data['repetition']), data)
			self.QUEUED_JOBS.append('%s_%d' % (data['job_id'], data['repetition']))

		else:
			self._print('found only trash results')
	


		file_logger = self.FILE_LOGGERS[0]
		file_logger.stop()
		del self.FILE_LOGGERS[0]

		if len(self.FILE_LOGGERS) > 0:
			file_logger = self.FILE_LOGGERS[0]
			file_logger.start()


#		if self.ROBOT_STATUS[exp_name] == 'usable':
#
#			# store data dict as attribute and append to processed jobs
#			setattr(self, 'info_dict_%s_%d' % (data['job_id'], data['repetition']), data)
#			self.QUEUED_JOBS.append('%s_%d' % (data['job_id'], data['repetition']))
#
#		else:
#			self._print('found only trash results')
#	
#		print('ROBOT QUEUED JOBS before', self.QUEUED_JOBS)


		# release bot
		if len(self.FILE_LOGGERS) == 0:
			bot = self.label_available(bot_name)
			for job in self.QUEUED_JOBS:
				self.PROCESSED_JOBS.append(job)
			self.QUEUED_JOBS = []
#
#		print('ROBOT QUEUED JOBS after', self.QUEUED_JOBS)
	




		
	def kill_running_robots(self, exp_identifier):
		self._print('killing parameter generation for %s' % exp_identifier)
		# just need to stop the thread
		self.ROBOT_STATUS[exp_identifier] = 'trash'

		








	def submit(self, bot_dict, experiment):
		self._print('submitting job %s to bot %s' % (experiment['job_id'], bot_dict['name']))

		# set bot to busy
		self.label_busy(bot_dict)
		experiment['bot_name'] = bot_dict['name']

		# find settings for the experiment to be run
		for exp_settings in self.settings['experiments']:
			if exp_settings['name'] == experiment['exp_identifier']:
				num_reps = exp_settings['repetitions']
				break

		# prepare and initialize the file logger
		for rep in range(num_reps):
			file_logger = FileLogger(action = self.process_evaluated_params, path = bot_dict['communication']['pick_up_path'])
			# if no file logger running, start this one
			if len(self.FILE_LOGGERS) == 0:
				file_logger.start()
			self.FILE_LOGGERS.append(file_logger)

			self.ROBOT_STATUS[experiment['exp_identifier']] = 'usable'

			# prepare and submit all possible experiemnts
			file_name = '%s/%s_rep_%d.pkl' % (self.settings['scratch_dir'], experiment['job_id'], rep)
			experiment['repetition'] = rep
			pickle.dump(experiment, open(file_name, 'wb'))

			# submit experiment to bot
			if 'host' in bot_dict['communication'].keys():
				# need to run scp
				subprocess.call('scp %s %s@%s:%s' % (file_name, bot_dict['communication']['username'], bot_dict['communication']['host'], bot_dict['communication']['dump_path']), shell = True)
			else:
				# can just copy file (e.g. to dropbox folder)
				shutil.copy2(file_name, bot_dict['communication']['dump_path'])

			# clean up
			os.remove(file_name)


	def boot_bots(self):
		for bot_dict in self.settings['bots']:
			status_file = bot_dict['communication']['status_file']
			if not os.path.isfile(status_file):
				data = {'status': 'running'}
			else:
				data = pickle.load(open(status_file, 'rb'))
				data['status'] = 'running'
			pickle.dump(data, open(status_file, 'wb'))



	def shutdown(self, bot_dict = None):

		# if no bot_dict, shutdown all robots
		if isinstance(bot_dict, dict):
			status_file = bot_dict['communication']['status_file']
			if not os.path.isfile(status_file):
				data = {'status': 'shutdown'}
			else:
				data = pickle.load(open(status_file, 'rb'))
				data['status'] = 'shutdown'
			pickle.dump(data, open(status_file, 'wb'))
		else:
			for bot_dict in self.settings['bots']:
				status_file = bot_dict['communication']['status_file']
				if not os.path.isfile(status_file):
					data = {'status': 'shutdown'}
				else:
					data = pickle.load(open(status_file, 'rb'))
					data['status'] = 'shutdown'
				pickle.dump(data, open(status_file, 'wb'))
