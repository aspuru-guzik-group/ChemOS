#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

from Communicator.SlackInterface.slack import SlackCommunicator	
from Utils.utils import Printer

#========================================================================

class Communicator(Printer, SlackCommunicator):

	RECEIVED_REQUESTS = []
	RECEIVED_FEEDBACK = []
					 

	def __init__(self, settings, verbose = True):
		Printer.__init__(self, 'COMMUNICATOR', color = 'grey')
		self.settings = settings
		self.account_details = self.settings['account_details']
		self.account_details['exp_names'] = [exp['name'] for exp in self.settings['experiments']]
		self.verbose  = verbose
		self.option   = self.settings['communicator']['type']
		if self.option == 'auto':
			# prepare file dumping mechanism for all experiments
			for experiment in self.settings['experiments']:
				request = {'exp_identifier': experiment['name']}
				self.RECEIVED_REQUESTS.append(request)

		elif self.option == 'gmail':
			# prepare gmail communication
			# GMailCommunicator.__init__(self.account_details)
			from Communicator.ChatBot.bot import Bot
			self.bot = Bot()
			raise NotImplementedError

		elif self.option == 'twitter':
			# prepare twitter communication
			# TwitterCommunicator.__init__(self.account_details)
			from Communicator.ChatBot.bot import Bot
			self.bot = Bot()
			raise NotImplementedError

		elif self.option == 'slack':
			from Communicator.ChatBot.bot import Bot
			self._print('setting up Slack streaming')
			self.bot = Bot()
			SlackCommunicator.__init__(self, self.account_details)

		else:
			self._print('did not understand option: %s' % self.option)
			raise NotImplementedError


	def _find_experimental_procedure(self, text):
		for exp_name in self.account_details['exp_names']:
			if exp_name in text:
				return exp_name
		else:
			return None




	def _process_request(self, author, request, kind = 'start'):
		# just store new requests locally as attributes to be picked up by chemOS

		exp_name = self._find_experimental_procedure(request.lower())
		if not exp_name:
			self._print('could not find request %s' % request)
			message = 'Could not find valid experiment identifier in message: {@FOUND_IDENT}.\nPlease choose your identifier from: {@EXP_IDENTS}'
			replace_dict = {'{@EXP_IDENTS}': ','.join(self.account_details['exp_names']),
							'{@FOUND_IDENT}': request}
			self.send_message(author, message, replace_dict)
			return None

		# parse the author
		if self.option == 'auto':
			request_author = {'contact': 'self'}
		elif self.option == 'gmail':
			request_author = {'contact': str(author)}
		elif self.option == 'slack':
			request_author = {'contact': str(author)}
		elif self.option == 'twitter':
			request_author = {}
			for prop in dir(author):
				if not callable(prop) and not prop.startswith('__'):
					att = getattr(author, prop)
					try:
						request_author[prop] = str(att)
					except TypeError:
						request_author[prop] = 'NONE'

		# construct request dictionary
		request = {'exp_identifier': exp_name, 'author': request_author, 'kind': kind}

		# store request dictionary
		self.RECEIVED_REQUESTS.append(request)

		return exp_name

	def _interpret_feedback(self, classification):
		self._print('WARNING: cannot interpret feedback yet!')
		return 0.



	def _process_feedback(self, author, classification):
		# interpret the feedback
		loss = self._interpret_feedback(classification)
		# construct the feedback dictionary
		feedback = {'loss': loss, 'author': author}
		# store feedback dictionary
		self.RECEIVED_FEEDBACK.append(feedback)




	def process_message(self, author, body):
		# use bot to classify message
		self._print('received message: %s | %s' % (author, body))
		classification = self.bot.get_classification(body)
		self._print('received classification: %s' % classification)


		if classification == 'start':

			exp_proced = self._process_request(author, body, 'start')
			response   = self.bot.response(body)
			replace_dict = {'{@EXP_PROCED}': exp_proced}
			if exp_proced:
				self.send_message(author, response, replace_dict)

		elif classification == 'restart':

			exp_proced = self._process_request(author, body, 'restart')
			response = self.bot.response(body)
			replace_dict = {'{@EXP_PROCED}': exp_proced}
			if exp_proced:
				self.send_message(author, response, replace_dict)

		elif classification == 'stop':
			
			exp_proced = self._process_request(author, body, 'stop')
			response   = self.bot.response(body)
			replace_dict = {'{@EXP_PROCED}': exp_proced}
			if exp_proced:
				self.send_message(author, response, replace_dict)


		elif classification == 'description_request':
	
			# find experimental procedure
			exp_proced   = self._find_experimental_procedure(body)
			# respond with description of procedure
			response     = self.bot.response(body)
			for exp in self.settings['experiments']:
				if exp['name'] == exp_proced: break
			replace_dict = {'{@EXP_DESCRIPTION}': exp['description']}
			self.send_message(author, response, replace_dict)


		elif classification == 'greeting':
			response = self.bot.response(body)
			self.send_message(author, response)


	def send_message(self, recipient, message, replace_dict = {}, **kwargs):
		for key, item in replace_dict.items():
			message = message.replace(str(key), str(item))
		self._send_message(recipient, message, **kwargs)




#		if classification == 'request':
#			# process the request
#			drink_name = self._process_request(author, body)
#			response   = self.bot.response(body)
#			self.send_message(author, response, {'{@DRINK_NAME}': drink_name}, subject = '[MargaritorBob] Order confirmation')
#
#		elif classification in ['one_star', 'two_star', 'three_star', 'four_star', 'five_star']:
#			# process the feedback
#			self._process_feedback(author, classification)
#			response = self.bot.response(body)
#			self.send_message(author, response, subject = '[MargaritorBob] Received your feedback!')



	def notify_user(self, info_dict):

		if self.option == 'auto':
			for experiment in self.settings['experiments']:
				# only request new experiments if we maxed out the repetitions
				if experiment['name'] == info_dict['exp_identifier'] and experiment['repetitions'] == info_dict['repetition'] + 1:
					request = {'exp_identifier': experiment['name'], 'kind': 'start'}
					self.RECEIVED_REQUESTS.append(request)

		elif self.option == 'slack':
			for experiment in self.settings['experiments']:
				# only request new experiments if we maxed out the repetitions
				if experiment['name'] == info_dict['exp_identifier'] and experiment['repetitions'] == info_dict['repetition'] + 1:
					request = {'exp_identifier': experiment['name'], 'kind': 'start'}
					self.RECEIVED_REQUESTS.append(request)



	def stream(self):
		print('starting stream')
		self._stream(self.process_message)


#========================================================================

