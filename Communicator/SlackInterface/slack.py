#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import os, json, flask
import pickle
import subprocess

from slackclient import SlackClient

from Utilities.file_logger import FileLogger

#========================================================================

class SlackCommunicator(object):

	def __init__(self, slack_settings, account_details):

		self.slack_settings  = slack_settings
		self.account_details = account_details
		self.client          = SlackClient(os.environ.get('SLACK_TOKEN'))


	def _process_message(self, *args, **kwargs):
		pass


	def _send_message(self, channel_id, message, file_names = None):
		info = self.client.api_call('chat.postMessage', channel = channel_id, text = message, username = 'chemos', icon_emoji = ':robot_face:', as_user = False)
		if isinstance(file_names, list):
			for file_name in file_names:
				with open(file_name, 'rb') as file_content:
					info = self.client.api_call('files.upload', channels = channel_id, file = io.BytesIO(file_content.read()), filename = file_name, title = 'Experiment progress', as_user = False, username = 'chemos')

	

	def _parse_message(self, file_name):
		if 'new_command' in file_name and 'pkl' in file_name:
			event = pickle.load(open(file_name, 'rb'))
			self._process_message(event['channel'], event['text'])
			os.remove(file_name)




	def _stream(self, process_message):
		self._process_message = process_message
		self.file_logger = FileLogger(action = self._parse_message, path = os.getcwd())
		self.file_logger.start()
		# pickle settings
		template = open('Communicator/SlackInterface/run_slack_stream.py', 'r').read()
		replace_dict = {'{@PORT}': self.slack_settings['port'], '{@CHANNEL_ID}': self.slack_settings['channel_id']}
		for key, item in replace_dict.items():
			template = template.replace(str(key), str(item))
		content = open('Communicator/SlackInterface/run_slack_stream.py', 'w')
		content.write(template)
		content.close()
		subprocess.call('python Communicator/SlackInterface/run_slack_stream.py', shell = True)


#========================================================================


