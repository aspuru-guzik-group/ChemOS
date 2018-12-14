#!/usr/bin/env python 

__author__ = 'Florian Hase'

#==================================================================

import os, json, flask
import pickle
import uuid

from slackclient import SlackClient

#==================================================================

app = flask.Flask(__name__)

#==================================================================

@app.route('/slack', methods = ['POST'])
def inbound():
	data     = flask.request.json
	response = flask.Response(json.dumps({key: data[key] for key in data.keys()}))

	try:
		event = data['event']
	except KeyError:
		return response, 200

#	print('****************')
#
#	for key, value in event.items():
#		print(key, '\t', value)
#
#	print('****************')

	if event['type'] == 'message':
		try:
			if event['subtype'] in ['bot_message', 'message_deleted']:
				return response, 200
			if 'UDMUX420Y' in event['user']:
				return response, 200
		except KeyError:
			pass

		channel_id = event['channel']
#		print('Channel ID', channel_id)

		if channel_id == 'CDQ3YC9DL':   #'C8YP312JD':
			# save event to pickle file
			file_name = 'new_command_%s.pkl' % str(uuid.uuid4())	
			pickle.dump(event, open(file_name, 'wb'))
		
	return response, 200

#==================================================================

if __name__ == '__main__':
	
	app.run(debug = True, port = 8000)

