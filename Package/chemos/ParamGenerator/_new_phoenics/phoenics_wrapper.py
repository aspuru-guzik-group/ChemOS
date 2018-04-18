#!/usr/bin/env python 

#============================================================================

import os
from ParamGenerator.optimizer_wrapper import OptimizerWrapper
from ParamGenerator.Phoenics.phoenics import Phoenics

#============================================================================
# Purpose of the wrapper: provide the computing environment for optimizer 
#						  instance such that multiple instance could be run 
#						  in parallel
#============================================================================

class PhoenicsWrapper(OptimizerWrapper):

	HOME     = os.getcwd() if not 'PHOENICS_HOME' in os.environ.keys() else os.environ['PHOENICS_HOME']
	TEMPLATE = open('%s/Templates/config_phoenics_template.dat' % HOME, 'r').read() 

	def __init__(self, settings, verbose = False):
		OptimizerWrapper.__init__(self, settings, verbose)


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
		file_name = self._write_config_file(content, 'phoenics')
		self.config_files[identifier] = file_name
		return file_name



	def get_instance(self, identifier, exp_settings):
		self.identifier   = identifier
		self.exp_settings = exp_settings
		self._set_config_file(identifier, exp_settings)

		# config file exists for sure; create optimizer instance
		# FIXME: potentially we need to pass more environmental variables to the optimizer instance
		phoenics = Phoenics(self.config_file)

		return phoenics

#============================================================================
