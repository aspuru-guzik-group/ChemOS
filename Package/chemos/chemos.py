#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import copy
import time
import numpy as np

from threading import Thread

from BotManager.bot_manager import BotManager
from Communicator.communicator import Communicator
from DatabaseHandler.feedback_handler import FeedbackHandler
from DatabaseHandler.request_handler import RequestHandler
from ParamGenerator.param_generator import ParamGenerator
from Utils import defaults
from Utils.utils import Printer

#========================================================================

np.set_printoptions(precision = 4)

#========================================================================

class ChemOS(Printer):	

	SETTINGS = defaults._DEFAULT_SETTINGS
	num_submitted_jobs = 0
	num_evaluated_jobs = 0

	def __init__(self, verbose = True):
		Printer.__init__(self, 'CHEM_OS', color = 'green')
		self.verbose = verbose

		# initialize bot manager
		if self.verbose: self._print('initializing bot manager')
		self.bot_manager = BotManager(settings = self.SETTINGS)
		# FIXME: THE LINE BELOW DO NOT ALIGN WITH THE IDEAL WORKFLOW
		self.bot_manager.boot_bots()

		# initialize communicator
		if self.verbose: self._print('initializing communicator')
		self.communicator = Communicator(settings = self.SETTINGS)

		# initialize feedback handler
		if self.verbose: self._print('initializing feedback handler')
		self.feedback_handler = FeedbackHandler(settings = self.SETTINGS)

		# initialize request handler
		if self.verbose: self._print('initializing request handler')
		self.request_handler = RequestHandler(settings = self.SETTINGS)


		# initialize parameter generator
		if self.verbose: self._print('initializing parameter generator')
		self.param_generator = ParamGenerator(settings = self.SETTINGS)


	def start_communication_stream(self):
		self._print('starting communication stream')
		self.stream_thread = Thread(target = self.communicator.stream)
		self.stream_thread.start()



	def purge(self, exp_identifier):
		# remove pending requests
		self.request_handler.remove_requests(exp_identifier)

		# remove collected feedback
		self.feedback_handler.remove_feedback(exp_identifier)


		# delete running parameter generations
		self.param_generator.kill_running_instances(exp_identifier)
		# delete running robots
		self.bot_manager.kill_running_robots(exp_identifier)
		# delete parameters in parameter database
		self.param_generator.remove_parameters(exp_identifier)





	def _run(self):

		# search for new requests
		new_requests = copy.deepcopy(self.communicator.RECEIVED_REQUESTS)
		for request in new_requests:
			if request['kind'] == 'start':
				self.request_handler.process_request(request)
			elif request['kind'] == 'restart':
				self.purge(request['exp_identifier'])
				self.request_handler.process_request(request)
			elif request['kind'] == 'stop':
				self.purge(request['exp_identifier'])

		for index in range(len(new_requests)):
			self.communicator.RECEIVED_REQUESTS.pop()

		# update list of new requests
		new_requests = self.request_handler.get_pending_request()
		if new_requests:

			# check if there are bots available for each new request
			for request in new_requests:
#				self._print('requesting available bot to process %s' % request['exp_identifier'])
				available_bot = self.bot_manager.get_available(request['exp_identifier'])
				if available_bot:

					# get parameters for this experiment
#					if self.verbose: self._print('requesting parameters for experiment %s' % request['exp_identifier'])

#					wait = True
#					while wait:
#						parameters, wait, retrain = self.param_generator.select_parameters(request['exp_identifier'])
#						time.sleep(2)
					parameters, wait, retrain = self.param_generator.select_parameters(request['exp_identifier'])
					if wait: continue

					request['parameters'] = parameters

					# now we have request, bot and parameters
					self.bot_manager.submit(available_bot, request)
					self.num_submitted_jobs += 1

					# change status of parameters
					self.request_handler.dump_parameters(request, parameters)
					self.request_handler.label_processing(request)

					# if the parameter database is exhausted we need to regenerate parameters
					if retrain:
						if self.verbose: self._print('parameter database exhausted for %s' % request['exp_identifier'])
						if not self.param_generator.BUSY[request['exp_identifier']]:

							# get observations for this experiment
							observations = self.feedback_handler.get_observations(request['exp_identifier'])

							if self.verbose: self._print('starting parameter generation process for %s' % request['exp_identifier'])
							self.param_generator.TARGET_SPECS[request['exp_identifier']] = observations
							self.param_generator.generate_new_parameters(request['exp_identifier'])


		# check for completed experiments
		processed_jobs = copy.deepcopy(self.bot_manager.PROCESSED_JOBS)
		for job_id in processed_jobs:
			info_dict = getattr(self.bot_manager, 'info_dict_%s' % job_id)
			self.feedback_handler.process_feedback(info_dict)
			self.communicator.notify_user(info_dict)
			self.bot_manager.PROCESSED_JOBS.remove(job_id)


		# check for new feedback
		exp_identifiers = self.feedback_handler.get_new_feedback()
#		if self.verbose and len(exp_identifiers) > 0: self._print('found new feedback')
		for exp_identifier in exp_identifiers:
			if not self.param_generator.BUSY[exp_identifier]:
				# get observations
				observations = self.feedback_handler.get_observations(exp_identifier)
				if self.verbose: self._print('starting parameter generation process for %s with %d observations' % (exp_identifier, len(observations)))
				# submit generation process
				self.param_generator.TARGET_SPECS[exp_identifier] = observations
				self.param_generator.generate_new_parameters(exp_identifier)
				# label feedback as used
				self.feedback_handler.set_all_to_used(exp_identifier)




#		pass
#	def _run_old(self):
#
#		# decision tree for new requests in request database - we try to process them!		
#		available_bot = self.job_manager.get_available()
#		if available_bot:
#			pending_request = self.request_handler.get_pending_request()
#			if pending_request:
#				# get recipe from database
#				if self.verbose: self._print('searching for parameters')
#
#				all_used = True
#				while all_used:# and not self.recipe_generator.BUSY:
#					self._print('waiting for parameter generator')
#					recipe, all_used = self.recipe_generator.select_recipe(pending_request['job_type'])
#					time.sleep(2)
#				print('GENERATED RECIPE:', recipe)
#				generate_new_recipes = all_used

#				# convert recipe based on bot spefications
#				expanded_recipe = self._expand_recipe(recipe, pending_request['job_type'], available_bot)
#				if self.verbose: self._print('submitting parameters to the bot')
#
#				# collect request, bot and recipe and submit job
#				print('EXPANDED RECIPE:', expanded_recipe)
#				self.job_manager.submit(available_bot, pending_request, expanded_recipe)				
#				self.num_submitted_jobs += 1
#
#				# change status of bot and recipe
#				self.job_manager.label_busy(available_bot)
#				self.request_handler.dump_recipe(pending_request, recipe)
#				self.request_handler.label_processing(pending_request)
#
#
#				# if the database is exhausted we need to re-generate recipes
#				if generate_new_recipes:
#					if self.verbose: self._print('parameter database exhausted')
#					if not self.recipe_generator.BUSY:
#						if self.verbose: self._print('starting parameter generation process')
#						self.recipe_generator.target_drink_name = pending_request['job_type']
#						self.start_recipe_generation(pending_request['job_type'])
#
#		# decision tree for newly received feedback - greedy retraining!
#		# check for new feedback
#		drink_name = self.feedback_handler.get_new_feedback()
#
#		if drink_name:
#			if self.verbose: self._print('found new feedback')
#			# start retraining
#			if not self.recipe_generator.BUSY:
#				if self.verbose: self._print('starting parameter generation process')
#				self.recipe_generator.target_drink_name = drink_name
#				self.start_recipe_generation(drink_name)
#				self.feedback_handler.set_all_to_used(drink_name)




	def run(self, max_iter = 10**12):

		if max_iter: self.max_iter = max_iter

		self.start_communication_stream()

		while self.num_submitted_jobs < self.max_iter:
			try:
				self._run()
				time.sleep(2)
			except (KeyboardInterrupt, SystemExit):
				self._print('shutting down ..')
				self._print('... turning off robots ...')
				self.bot_manager.shutdown()
				self._print('... completed shutdown')
				break
			self.num_submitted_jobs = -10
		else:
			self._print('completed all experiments: ')
			self._print('shutting down ..')
			self._print('... turning off robots ...')
			self.bot_manager.shutdown()
			self._print('... completed shutdown')

		self._print('good night!')

#========================================================================

'''
settings ...
--> algorithm: dictionary with optimization algorithm parameters 
--> param_database: dictionary, needs path and type
'''

#========================================================================

if __name__ == '__main__':

	try:
		max_iter = int(sys.argv[1])
	except IndexError:
		max_iter = 10**12
	chem_os  = ChemOS()
	chem_os.run(max_iter)

