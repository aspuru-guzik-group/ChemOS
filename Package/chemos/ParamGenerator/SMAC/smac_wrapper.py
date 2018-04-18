#!/usr/bin/env python 

__author__ = 'Florian Hase'

#============================================================================

import os
from ParamGenerator.SMAC.smac import SMAC
from Utils.utils import Replacable

#============================================================================

class SmacWrapper(Replacable):

	HOME     = os.getcwd() if not 'SMAC_HOME' in os.environ.keys() else os.environ['SMAC_HOME']
	TEMPLATE = open('%s/Templates/config_smac_template.dat' % HOME, 'r').read() 


	def __init__(self, settings, verbose = False):
		Replacable.__init__(self)
		self.settings     = settings
		self.config_files = {}



	def _write_config_file(self, content):
		config_name  = '%s/.config_smac_%s.dat' % (self.settings['scratch_dir'], self.identifier)
		file_content = open(config_name, 'w')
		file_content.write(content)
		file_content.close()
		return config_name




	def _create_config_file(self, identifier, exp_settings):
		# get general parameters
		replace_dict = {'{@NUM_BATCHES}': self.settings['num_batches'],
						'{@BATCH_SIZE}': self.settings['batch_size'],}
		# create the variable list - we need the experiment specifications
		new_variable = ''
		for variable in exp_settings['variables']:
			new_variable += '{"%s": {' % variable['name']
			new_variable += ' "low": 0.0, '
			new_variable += ' "high": 1.0, '
			new_variable += ' "type": "%s", ' % variable['type']
			new_variable += ' "size": %s} }' % variable['size']
			new_variable += ', '
		new_variable = new_variable[:-2]
		replace_dict['{@VARIABLE_LIST}'] = new_variable
		content = self.TEMPLATE
		for key, value in replace_dict.items():
			content = content.replace(str(key), str(value))
		# write config file
		file_name = self._write_config_file(content)
		self.config_files[identifier] = file_name
		return file_name




	def _set_config_file(self, identifier, exp_settings):
		self.identifier = identifier
		# check if config file exists
		if not self.identifier in self.config_files.keys():
			self._create_config_file(identifier, exp_settings)
		self.config_file = self.config_files[self.identifier]


	def get_instance(self, identifier, exp_settings):
		self.identifier   = identifier
		self.exp_settings = exp_settings
		self._set_config_file(identifier, exp_settings)

		smac = SMAC(self.config_file, self.settings['scratch_dir'])
		return smac
