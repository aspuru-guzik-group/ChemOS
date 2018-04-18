#!/usr/bin/env python 

__author__ = 'Florian Hase'

#============================================================================

from Utils.utils import Replacable

#============================================================================

class OptimizerWrapper(Replacable):

	def __init__(self, settings, verbose = False):
		Replacable.__init__(self)
		self.settings     = settings
		self.config_files = {}


	def _write_config_file(self, content, key):
		if key == 'spearmint':
			config_name = '%s/config.json' % (self.settings['scratch_dir'])
		else:
			config_name  = '%s/.config_%s_%s.dat' % (self.settings['scratch_dir'], key, self.identifier)
		file_content = open(config_name, 'w')
		file_content.write(content)
		file_content.close()
		return config_name


	def _set_config_file(self, identifier, exp_settings):
		self.identifier = identifier
		# check if config file exists
		if not self.identifier in self.config_files.keys():
			self._create_config_file(identifier, exp_settings)
		self.config_file = self.config_files[self.identifier]
