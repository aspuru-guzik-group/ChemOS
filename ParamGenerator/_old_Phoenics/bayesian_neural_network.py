#!/usr/bin/env python

import numpy as np 
import theano 
import theano.tensor as T 
import pymc3 as pm 

theano.config.compute_test_value = 'ignore'

#================================================================================================================

HIDDEN_SHAPE = 12
SIGMA        = 1.

NUM_SAMPLES  = 1500
BURNIN       = 1000
THINNING     = 10

theano.compiledir = '{@COMPILE_DIR}'
theano.base_compiledir = '{@BASE_COMPILE_DIR}'

#================================================================================================================

class ProbDist(object):

	def __init__(self, observed, losses, batches):
		self.observed = observed
		self.losses   = losses
		self.batches  = batches
#		print('# LOG | ... ... OPTIMIZER ... LOSSES: ', self.losses)
		self.num_obs  = len(observed)
		self.characteristic_distance = 1. / float(np.sqrt(self.num_obs))
		if self.num_obs > 0:
			self.in_shape  = observed.shape[1]
			self.out_shape = observed.shape[1]
		self.burnin   = BURNIN
		self.thinning = THINNING 


	def _gauss(self, x, mu, sd):
		return np.exp( - (x - mu)**2 / (2. * sd**2)) / np.sqrt(2. * np.pi * sd**2)


	def create_model(self):

		with pm.Model() as self.model:

			self.w0_mu = pm.Normal('w0_mu', 0., sd = SIGMA, shape = (self.in_shape, HIDDEN_SHAPE))
			self.w1_mu = pm.Normal('w1_mu', 0., sd = SIGMA, shape = (HIDDEN_SHAPE, HIDDEN_SHAPE))
			self.w2_mu = pm.Normal('w2_mu', 0., sd = SIGMA, shape = (HIDDEN_SHAPE, self.out_shape))
			self.b0_mu = pm.Normal('b0_mu', 0., sd = SIGMA, shape = (HIDDEN_SHAPE))
			self.b1_mu = pm.Normal('b1_mu', 0., sd = SIGMA, shape = (HIDDEN_SHAPE))
			self.b2_mu = pm.Normal('b2_mu', 0., sd = SIGMA, shape = (self.out_shape))

			self.mu1 = pm.Deterministic('mu_1', pm.math.tanh(pm.math.dot(self.observed, self.w0_mu) + self.b0_mu))
			self.mu2 = pm.Deterministic('mu_2', pm.math.tanh(pm.math.dot(self.mu1, self.w1_mu) + self.b1_mu))
			self.mu3 = pm.Deterministic('mean', 1.2 * pm.math.sigmoid(pm.math.dot(self.mu2, self.w2_mu) + self.b2_mu) - 0.1)
			self.tau = pm.Gamma('tau',  12 * self.num_obs, 1., shape = (self.num_obs, self.out_shape))

			self.sd  = pm.Deterministic('sd', 1 / T.sqrt(self.tau))
			self.out = pm.Normal('out', self.mu3, tau = self.tau, observed = self.observed)




	@theano.configparser.change_flags(compute_test_value='ignore')
	def sample(self, num_samples = NUM_SAMPLES):
		self.num_samples = num_samples
		with self.model:
			theano.config.compute_test_value = 'ignore'
#			inference = pm.ADVI()
			approx = pm.fit(n = 2 * 5 * 10**4, obj_optimizer = pm.adam(learning_rate = 0.1))
			self.trace = approx.sample(draws = 10**4)
			del approx


	def sample_posterior(self, num_samples = 1500):
		with self.model:
			self.ppc = pm.sample_ppc(self.trace, samples = num_samples)
		return self.ppc['out']


	def build_penalty(self):
		self.clean_samples = int((self.num_samples - self.burnin) / self.thinning)

		self.mus = self.trace['mean'][self.burnin::self.thinning].copy()
		self.sds = self.trace['sd'][self.burnin::self.thinning].copy()
	
		del self.trace
	
		if self.batches == 1:
			self.lambda_values = [0.]
		else:
			self.lambda_values = np.linspace(-0.2, 0.8, self.batches)


		# no theano computation
		self.prob  = lambda x, mu, sd: np.prod(self._gauss(x, mu, sd), axis = 1)
		self.probs = lambda x: np.mean( [self.prob(x, self.mus[i], self.sds[i]) for i in range(self.clean_samples)], axis = 0 )
		
		def penalty(x, index):
			probs_x = self.probs(x)
			result  = (np.dot(self.losses, probs_x) + self.lambda_values[index]) / (np.sum(probs_x) + 1.)
			return result

		self.get_probs = self.probs
		self.get_dot_prods = lambda x: np.dot(self.losses, x)
		self.penalty = penalty

		self.penalties = [lambda x: self.penalty(x, index) for index in range(len(self.lambda_values))]

#================================================================================================================


if __name__ == '__main__sdf':
	data = corner_3d()

	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')

	ax.scatter(data[:, 0], data[:, 1], data[:, 2])
	plt.show()
