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

#		print(observations)
		
		# get all the losses
		losses = {}
		for element in observations:
			elem_losses = element['loss']
			for key, value in elem_losses.items():
				if not key in losses:
					losses[key] = []
				losses[key].append(value)

		plt.clf()

		for key, loss_list in losses.items():
			domain = np.arange(len(loss_list)) + 1
			loss_list = np.array(loss_list)
	
#			plt.plot(domain, loss_list, ls = '', marker = 'o', color = 'w', markersize = 10)
#			plt.plot(domain, loss_list, ls = '', marker = 'o', color = 'k', alpha = 0.8, markersize = 10)
			plt.plot(domain, loss_list, ls = '', marker = 'o', alpha = 0.5, markersize = 10, label = key)
			
		plt.legend()
		plt.xlabel('# experiments')

		file_name = '%s/%s.png' % (self.settings['scratch_dir'], str(uuid.uuid4()))
		plt.savefig(file_name, bbox_to_inches = 'tight')

		exp_dict = {'request_details': request, 'observations': observations, 'progress_file': file_name, 'status': 'new'}
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
