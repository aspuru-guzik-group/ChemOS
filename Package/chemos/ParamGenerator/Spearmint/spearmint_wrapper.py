#!/usr/bin/env python 

__author__ = 'Florian Hase'

#============================================================================

import os

from ParamGenerator.optimizer_wrapper import OptimizerWrapper
from ParamGenerator.Spearmint.spearmint_instance import Spearmint

#============================================================================

class SpearmintWrapper(OptimizerWrapper):

	HOME = os.getcwd() if not 'SPEARMINT_HOME' in os.environ.keys() else os.environ['SPEARMINT_HOME']
	TEMPLATE = open('%s/Templates/config_spearmint_template.json' % HOME, 'r').read() 

	def __init__(self, settings, verbose = False):
		OptimizerWrapper.__init__(self, settings, verbose)


	def _create_config_file(self, identifier, exp_settings):
		replace_dict = {'{@BATCH_SIZE}': self.settings['batch_size']}
		# create the variable list - we need the experiment specifications
		new_variable = ''
		for variable in exp_settings['variables']:
			new_variable += '"%s": {' % variable['name']
			new_variable += ' "min": 0.0, '
			new_variable += ' "max": 1.0, '
			new_variable += ' "type": "%s", ' % variable['type']
			new_variable += ' "size": %s}' % variable['size']
			new_variable += ', '
		new_variable = new_variable[:-2]
		replace_dict['{@VARIABLE_LIST}'] = new_variable
		content = self.TEMPLATE
		for key, value in replace_dict.items():
			content = content.replace(str(key), str(value))
		# write config file
		file_name = self._write_config_file(content, 'spearmint')
		self.config_files[identifier] = file_name
		return file_name



	def get_instance(self, identifier, exp_settings):
		self.identifier   = identifier 
		self.exp_settings = exp_settings
		self._set_config_file(identifier, exp_settings)

		spearmint = Spearmint(self.config_file, self.settings['scratch_dir'])

		return spearmint

#============================================================================
