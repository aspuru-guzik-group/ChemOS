#!/usr/bin/env python 

import json 
import pickle

#=======================================================================

def pickle_load(file_name):
	return pickle.load(open(file_name, 'rb'))

def pickle_dump(dump_dict, file_name):
	pickle.dump(dump_dict, open(file_name, 'wb'))

#=======================================================================

class ParserJSON(object):

	def __init__(self, file_name = None):
		self.file_name = file_name


#	def _clean(self):
#		self.param_dict = {}
#		for param in self.json:
#			param_settings = {'restricted': False}
#			for key, item in self.json[param].items():
#				if type(item) in [int, float]:
#					param_settings[str(key)] = item
#				else:
#					if key == 'restricted':
#						param_settings[str(key)] = item == 'yes'
#					else:
#						param_settings[str(key)] = str(item)
#			self.param_dict[param] = param_settings


	def parse(self, file_name = None):
		if file_name:
			self.json = json.loads(file_name).read()
		else:
			self.json = json.loads(open(self.file_name).read())
		self.param_dict = self.json


#=======================================================================

if __name__ == '__main__':
	parser = ParserJSON('config.txt')
	parser.parse()
	print(parser.param_dict)
	quit()
