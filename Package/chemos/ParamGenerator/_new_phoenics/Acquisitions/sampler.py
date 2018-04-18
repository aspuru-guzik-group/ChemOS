#!/usr/bin/env python

__author__ = 'Florian Hase'

#========================================================================

import numpy as np 
from scipy.optimize import minimize
from multiprocessing import Process, Queue

from Acquisitions.optimizer import ParameterOptimizer
from RandomNumberGenerator.random_number_generator import RandomNumberGenerator
from PhoenicsUtils.utils import VarDictParser

#========================================================================

class AcquisitionFunctionSampler(VarDictParser):

	def __init__(self, var_infos, var_dicts):
		VarDictParser.__init__(self, var_dicts)

		self.local_opt = ParameterOptimizer(var_dicts)

		self.var_infos = var_infos
		for key, value in self.var_infos.items():
			setattr(self, str(key), value)
		self.total_size = np.sum(self.var_sizes)
		self.random_number_generator = RandomNumberGenerator()




	def _get_random_proposals(self, current_best, num_samples):
		# get uniform samples first
		uniform_samples = []
		for var_index, full_var_dict in enumerate(self.var_dicts):
			var_dict = full_var_dict[self.var_names[var_index]]
			sampled_values = self.random_number_generator.generate(var_dict, size = (self.var_sizes[var_index], self.total_size * num_samples))
			uniform_samples.extend(sampled_values)
		uniform_samples = np.array(uniform_samples).transpose()
		proposals = uniform_samples
		return np.array(proposals)



	def _gen_set_to_zero_vector(self, sample):
		vector = np.ones(len(sample))
		start_index = 0
		for var_index, keep_num in enumerate(self.var_keep_num):
			var_size = self.var_sizes[var_index]
			indices  = np.arange(var_size)
			np.random.shuffle(indices)
			vector[ start_index + indices[:var_size - keep_num] ] = 0.
			start_index += indices
		return vector



	def _proposal_optimization_thread(self, batch_index, queue = None):
		print('starting process for ', batch_index)
		# prepare penalty function
		def penalty(x):
			num, den = self.penalty_contributions(x)
			return (num + self.lambda_values[batch_index]) / den

		optimized = []
#		for sample in self.proposals:
#			if np.random.uniform() < 0.5:
#				optimized.append(sample)
#				continue
#			res = minimize(penalty, sample, method = 'L-BFGS-B', options = {'maxiter': 25})
#
#			# FIXME
#			if np.any(res.x < self.var_lows) or np.any(res.x > self.var_highs):
#				optimized.append(sample)
#			else:
#				optimized.append(res.x)
	
		for sample in self.proposals:

			# set some entries to zero!
			set_to_zero = self._gen_set_to_zero_vector(sample)
			nulls = np.where(set_to_zero == 0)[0]

			opt = self.local_opt.optimize(penalty, sample * set_to_zero, max_iter = 10, ignore = nulls)

			optimized.append(opt)


		optimized = np.array(optimized)
		optimized[:, self._ints] = np.around(optimized[:, self._ints])
		optimized[:, self._cats] = np.around(optimized[:, self._cats])

		print('finished process for ', batch_index)
		if queue:
			queue.put({batch_index: optimized})
		else:
			return optimized



	def _optimize_proposals(self, proposals, parallel):
		self.proposals = proposals

		if parallel == 'True':
			q = Queue()
			processes = []
			for batch_index in range(len(self.lambda_values)):
				process = Process(target = self._proposal_optimization_thread, args = (batch_index, q))
				processes.append(process)
				process.start()

			for process_index, process in enumerate(processes):
				process.join()

			print('got all results')

			result_dict = {}
			while not q.empty():
				results = q.get()
				for key, value in results.items():
					result_dict[key] = value

		elif parallel == 'False':
			result_dict = {}
			for batch_index in range(len(self.lambda_values)):
				result_dict[batch_index] = self._proposal_optimization_thread(batch_index)

		else:
			raise NotImplementedError()

		samples = [result_dict[batch_index] for batch_index in range(len(self.lambda_values))]
		return np.array(samples)




	def sample(self, current_best, penalty_contributions, lambda_values, num_samples = 10, parallel = 'True'):

		self.penalty_contributions = penalty_contributions
		self.lambda_values         = lambda_values

		# FIXME samples are not yet optimized!
		proposals = self._get_random_proposals(current_best, num_samples)
		print('# ... optimizing')
		proposals = self._optimize_proposals(proposals, parallel)
		return proposals

#========================================================================
