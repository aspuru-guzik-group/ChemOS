#!/usr/bin/env python

__author__ = 'Florian Hase'

#========================================================================

import os, uuid
import numpy as np 

import matplotlib as mpl 
mpl.use('Agg')
import matplotlib.pyplot as plt

from threading import Thread

from Utilities.file_logger import FileLogger
from Utilities.misc import Printer

#========================================================================

class Analyzer(Printer):

	ANALYZED_EXPERIMENTS = []

	def __init__(self, settings, verbose = True):

		Printer.__init__(self, 'ANALYZER', color = 'grey')
		self.settings = settings
		self.verbose = verbose


	def _analyze(self, request, observations):
		# get all the losses
		losses = []
		for element in observations:
			losses.append(element[self.settings['loss_name']])
		domain = np.arange(len(losses)) + 1
		losses = np.array(losses)

		plt.plot(domain, losses, ls = '', marker = 'o', color = 'w', markersize = 10)
		plt.plot(domain, losses, ls = '', marker = 'o', color = 'k', alpha = 0.8, markersize = 10)
		plt.plot(domain, losses, ls = '', marker = 'o', color = 'w', alpha = 0.5, markersize = 10)
		plt.xlabel('# experiments')
		plt.ylabel('%s' % self.settings['loss_name'])

		file_name = '%s/%s.png' % (self.settings['scratch_dir'], str(uuid.uuid4()))
		plt.savefig(file_name, bbox_to_inches = 'tight')

		exp_dict = {'request': request, 'observations': observations, 'file_name': file_name, 'status': 'new'}
		self.ANALYZED_EXPERIMENTS.append(exp_dict)



	def analyze(self, request, observations):
		analysis_thread = Thread(target = self._analyze, args = (request, observations))
		analysis_thread.start()		



	def get_analyzed_experiments(self):
		analyzed_experiments = []
		for exp_dict in self.ANALYZED_EXPERIMENTS:
			if exp_dict['status'] == 'new':
				analyzed_experiments.append(exp_dict)
				exp_dict['status'] = 'fetched'
		return analyzed_experiments