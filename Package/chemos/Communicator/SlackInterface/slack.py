#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import os, json, flask
import pickle
import subprocess

from slackclient import SlackClient

from Utils.file_logger import FileLogger

#========================================================================

class SlackCommunicator(object):

	def __init__(self, account_details):

		self.account_details = account_details
		self.client          = SlackClient(os.environ.get('SLACK_TOKEN'))


	def _process_message(self, *args, **kwargs):
		pass


	def _send_message(self, channel_id, message):
		info = self.client.api_call('chat.postMessage', channel = channel_id, text = message, username = 'chemos', icon_emoji = ':robot_face:', as_user = False)



	def _parse_message(self, file_name):
		if 'new_command' in file_name and 'pkl' in file_name:
			event = pickle.load(open(file_name, 'rb'))
			self._process_message(event['channel'], event['text'])
			os.remove(file_name)




	def _stream(self, process_message):
		self._process_message = process_message
		self.file_logger = FileLogger(action = self._parse_message, path = os.getcwd())
		self.file_logger.start()
		subprocess.call('python Communicator/SlackInterface/run_slack_stream.py', shell = True)


#========================================================================


