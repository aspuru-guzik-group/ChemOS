#!/usr/bin/env python 

__author__ = 'Florian Hase'

#========================================================================

import json
from lazyme.string import color_print

#========================================================================

class Printer(object):

	def __init__(self, template, color = 'blue'):
		
		self.color    = color
		self.template = template


	def _print(self, message):
		to_print = self.template + ' ... ' + message + ' ... '
		color_print(to_print, color = self.color, bold = True)

#========================================================================

class Replacable(object):

	def __init__(self):
		pass

	def _replace_all(self, string, replace_dict):
		for key, value in replace_dict.items():
			string = string.replace(str(key), str(value))
		return string

#========================================================================

class ParserJSON(object):

	def __init__(self, file_name = None):
		self.file_name = file_name


	def parse(self, file_name = None):
		if file_name:
			self.json = json.loads(file_name).read()
		else:
			self.json = json.loads(open(self.file_name).read())
		self.param_dict = self.json

#========================================================================
