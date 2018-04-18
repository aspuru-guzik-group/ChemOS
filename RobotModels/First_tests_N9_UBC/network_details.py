#!/usr/bin/env python 

import pickle

import numpy as np 
import tensorflow as tf
import edward as ed

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#======================================================================================

class ManagerDummy(object):

	def __init__(self):
		self.mean_features = None 
		self.std_features  = None 
		self.mean_targets  = None 
		self.std_targets   = None

		self.train_features = None
		self.train_targets  = None

#======================================================================================

class ClippedAdamOptimizer(tf.train.AdamOptimizer):
    """
    Clipped version adam optimizer, where its gradient is clipped by value
    so that it cannot be too large.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-08, use_locking=False,
                 clip_func=lambda x: tf.clip_by_value(x, 1.0e-4, 1.0e4),
                 name='Adam'):
        super(ClippedAdamOptimizer, self).__init__(
            learning_rate, beta1, beta2, epsilon, use_locking, name)
        self._clip_func = clip_func

    def compute_gradients(self, loss, var_list=None, gate_gradients=None,
                          aggregation_method=None,
                          colocate_gradients_with_ops=False, grad_loss=None):
        grad_and_vars = super(ClippedAdamOptimizer, self).compute_gradients(
            loss, var_list, gate_gradients, aggregation_method,
            colocate_gradients_with_ops, grad_loss)
        # clip func
        if self._clip_func is None:
            return grad_and_vars
        return [(self._clip_func(g, v) if g is not None else (g, v)
                for g, v in grad_and_vars)]

#======================================================================================


class BayesianNeuralNetwork(object):

	BATCH_SIZE    = 100
	IN_SHAPE      = 6
	OUT_SHAPE     = 1
	HIDDEN_SHAPE  = 36
	NUM_LAYERS    = 3
	ACT_FUNC      = 'leaky_relu'
	LEARNING_RATE = 10**(-2.00) 


	def generator(self, arrays, batch_size):
		starts = [0] * len(arrays)
		while True:
			batches = []
			for i, array in enumerate(arrays):
				start = starts[i]
				stop  = start + batch_size
				diff  = stop - array.shape[0]
				if diff <= 0:
					batch = array[start: stop]
					starts[i] += batch_size
				else:
					batch = np.concatenate((array[start:], array[:diff]))
					starts[i] = diff
				batches.append(batch)
			yield batches


	def __init__(self):#, data_file):
		self.manager = ManagerDummy()
		self.train_features = None
		self.train_targets  = None
		self.valid_features = None 
		self.valid_targets  = None




	def _prepare_network(self):

		# get batched training set
		self.batched_train_data = self.generator([self.manager.train_features, self.manager.train_targets], self.BATCH_SIZE)

		# get weights and biases
		self.weight_shapes = [(self.IN_SHAPE, self.HIDDEN_SHAPE)]
		self.bias_shapes   = [(self.HIDDEN_SHAPE)]
		for layer_index in range(1, self.NUM_LAYERS - 1):
			self.weight_shapes.append((self.HIDDEN_SHAPE, self.HIDDEN_SHAPE))
			self.bias_shapes.append((self.HIDDEN_SHAPE))
		self.weight_shapes.append((self.HIDDEN_SHAPE, self.OUT_SHAPE))
		self.bias_shapes.append((self.OUT_SHAPE))

		# get activation functions
		tf_activation_functions = {'softsign': tf.nn.softsign, 'softplus': tf.nn.softplus, 'relu': tf.nn.relu6, 'tanh': tf.nn.tanh, 
								   'leaky_relu': lambda x: tf.minimum(tf.maximum(x, -0.1 * x), 10**4)}
		np_activation_functions = {'softsign': lambda x: x / (1. + np.abs(x)), 'softplus': lambda x: np.log(np.exp(x) + 1), 'relu': lambda x: np.minimum(np.maximum(x, 0.), 6.), 'tanh': lambda x: np.tanh(x), 
								   'leaky_relu': lambda x: np.minimum(np.maximum(x, -0.1 * x), 10**4)}
		self.tf_activation_functions = {'softsign': tf.nn.softsign, 'softplus': tf.nn.softplus, 'relu': tf.nn.relu6, 'tanh': tf.nn.tanh, 
								   'leaky_relu': lambda x: tf.minimum(tf.maximum(x, -0.1 * x), 10**4)}
		self.np_activation_functions = {'softsign': lambda x: x / (1. + np.abs(x)), 'softplus': lambda x: np.log(np.exp(x) + 1), 'relu': lambda x: np.minimum(np.maximum(x, 0.), 6.), 'tanh': lambda x: np.tanh(x), 
								   'leaky_relu': lambda x: np.minimum(np.maximum(x, -0.1 * x), 10**4)}
		self.act_tf = [tf_activation_functions[self.ACT_FUNC] for i in range(self.NUM_LAYERS)]
		self.act_np = [np_activation_functions[self.ACT_FUNC] for i in range(self.NUM_LAYERS)]



	def construct_networks(self):
		print('... constructing network')
		self._prepare_network()
		self.x = tf.placeholder(tf.float32, shape = (None, self.IN_SHAPE))
		self.y = tf.placeholder(tf.float32, shape = (None, self.OUT_SHAPE))

		# initialize weights and biases
		for layer_index in range(self.NUM_LAYERS):
			if layer_index == 0:
				setattr(self, 'reg_w%d' % layer_index, ed.models.Exponential(rate = tf.nn.softplus(tf.ones(self.weight_shapes[layer_index]))))
				setattr(self, 'reg_b%d' % layer_index, ed.models.Exponential(rate = tf.nn.softplus(tf.ones(self.bias_shapes[layer_index]))))
				setattr(self, 'w%d' % layer_index, ed.models.Laplace(loc = tf.zeros(self.weight_shapes[layer_index]), scale = getattr(self, 'reg_w%d' % layer_index)))
				setattr(self, 'b%d' % layer_index, ed.models.Laplace(loc = tf.zeros(self.bias_shapes[layer_index]), scale = getattr(self, 'reg_b%d' % layer_index)))
			else:
				setattr(self, 'w%d' % layer_index, ed.models.Laplace(loc = tf.zeros(self.weight_shapes[layer_index]), scale = tf.nn.softplus(tf.ones(self.weight_shapes[layer_index]))))
				setattr(self, 'b%d' % layer_index, ed.models.Laplace(loc = tf.zeros(self.bias_shapes[layer_index]), scale = tf.nn.softplus(tf.ones(self.bias_shapes[layer_index]))))


		# construct network graph
		self.fc0 = self.act_tf[0](tf.matmul(self.x, self.w0) + self.b0)
		for layer_index in range(1, self.NUM_LAYERS - 1):
			setattr(self, 'fc%d' % layer_index, self.act_tf[layer_index](tf.matmul(getattr(self, 'fc%d' % (layer_index - 1)), getattr(self, 'w%d' % layer_index)) + getattr(self, 'b%d' % layer_index)))
		layer_index += 1
		setattr(self, 'fc%d' % layer_index, tf.matmul(getattr(self, 'fc%d' % (layer_index - 1)), getattr(self, 'w%d' % layer_index)) + getattr(self, 'b%d' % layer_index))
		self.net_out = getattr(self, 'fc%d' % layer_index)
		self.net_put = tf.nn.relu(self.net_out)

		self.output = ed.models.Normal(loc = self.net_out, scale = 10**(-4.0))		

		# inference
		for layer_index in range(self.NUM_LAYERS):
			reg_w = ed.models.Exponential(tf.nn.softplus(tf.Variable(tf.zeros(self.weight_shapes[layer_index]))))
			reg_b = ed.models.Exponential(tf.nn.softplus(tf.Variable(tf.zeros(self.bias_shapes[layer_index]))))
			setattr(self, 'q_reg_w%d' % layer_index, reg_w)
			setattr(self, 'q_reg_b%d' % layer_index, reg_b)
			setattr(self, 'q_w%d' % layer_index, ed.models.Laplace(loc = tf.Variable(tf.zeros(self.weight_shapes[layer_index])), scale = reg_w))
			setattr(self, 'q_b%d' % layer_index, ed.models.Laplace(loc = tf.Variable(tf.zeros(self.bias_shapes[layer_index])), scale = reg_b))

		# set up training graph
		var_dict = {}
		for layer_index in range(self.NUM_LAYERS):
			var_dict[getattr(self, 'w%d' % layer_index)] = getattr(self, 'q_w%d' % layer_index)
			var_dict[getattr(self, 'b%d' % layer_index)] = getattr(self, 'q_b%d' % layer_index)
	
		self.inference = ed.KLqp(var_dict, data = {self.output: self.y})
		optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
		self.inference.initialize(optimizer = optimizer)#, scale = {self.output: len(self.train_features) / self.BATCH_SIZE}, n_samples = 5)
		tf.global_variables_initializer().run()


	def train(self, train_iters, epoch = -1):
		for i in range(train_iters):
			x_batch, y_batch = next(self.batched_train_data)
			self.inference.update(feed_dict = {self.x: x_batch, self.y: y_batch})	



	def _get_posterior(self, n_post = 200):
#		print('# sampling posterior ...')
		self.post_vars = {}
		for layer_index in range(self.NUM_LAYERS):
			self.post_vars['post_w%d' % layer_index] = getattr(self, 'q_w%d' % layer_index).sample(n_post).eval()
			self.post_vars['post_b%d' % layer_index] = getattr(self, 'q_b%d' % layer_index).sample(n_post).eval()
		self.has_posterior = True



	def _predict(self, n_post = 50):
		post_vars = {}
		for layer_index in range(self.NUM_LAYERS):
			post_vars['post_w%d' % layer_index] = getattr(self, 'q_w%d' % layer_index).sample(n_post).eval()
			post_vars['post_b%d' % layer_index] = getattr(self, 'q_b%d' % layer_index).sample(n_post).eval()

		# predict for training set
		train_samples = []
		for n_sample in range(n_post):
			y_sampled = np.dot(self.train_features, post_vars['post_w0'][n_sample]) + post_vars['post_b0'][n_sample]
			y_sampled = self.act_np[0](y_sampled)

			for layer_index in range(1, self.NUM_LAYERS - 1):
				y_sampled = np.dot(y_sampled, post_vars['post_w%d' % layer_index][n_sample]) + post_vars['post_b%d' % layer_index][n_sample]
				y_sampled = self.act_np[layer_index](y_sampled)
			layer_index += 1
			y_sampled = np.dot(y_sampled, post_vars['post_w%d' % layer_index][n_sample]) + post_vars['post_b%d' % layer_index][n_sample]
			y_sampled = self.np_activation_functions['relu'](y_sampled)
			train_samples.append(y_sampled)

		valid_samples = []
		for n_sample in range(n_post):
			y_sampled = np.dot(self.valid_features, post_vars['post_w0'][n_sample]) + post_vars['post_b0'][n_sample]
			y_sampled = self.act_np[0](y_sampled)
			for layer_index in range(1, self.NUM_LAYERS - 1):
				y_sampled = np.dot(y_sampled, post_vars['post_w%d' % layer_index][n_sample]) + post_vars['post_b%d' % layer_index][n_sample]
				y_sampled = self.act_np[layer_index](y_sampled)
			layer_index += 1
			y_sampled = np.dot(y_sampled, post_vars['post_w%d' % layer_index][n_sample]) + post_vars['post_b%d' % layer_index][n_sample]
			y_sampled = self.np_activation_functions['relu'](y_sampled)
			valid_samples.append(y_sampled)

		return np.array(train_samples), np.array(valid_samples), post_vars




	def predict(self, params, n_post = 200, rescale = True):

		# TODO: NEEDS TO BE CHANGED!!!
		self._get_posterior(n_post)


		# rescale normal mode coords
		if rescale:
			features = (params - self.manager.mean_features) / self.manager.std_features
		else: 
			features = params

		# predict dissociation times
		time_samples = []
		for n_sample in range(len(self.post_vars['post_w0'])):
			y_sampled = np.dot(features, self.post_vars['post_w0'][n_sample]) + self.post_vars['post_b0'][n_sample]
			y_sampled = self.act_np[0](y_sampled)
			for layer_index in range(1, self.NUM_LAYERS - 1):
				y_sampled = np.dot(y_sampled, self.post_vars['post_w%d' % layer_index][n_sample]) + self.post_vars['post_b%d' % layer_index][n_sample]
				y_sampled = self.act_np[layer_index](y_sampled)
			layer_index += 1
			y_sampled = np.dot(y_sampled, self.post_vars['post_w%d' % layer_index][n_sample]) + self.post_vars['post_b%d' % layer_index][n_sample]
			y_sampled = self.np_activation_functions['relu'](y_sampled)
			time_samples.append(y_sampled)
		time_samples = np.array(time_samples)

		# rescale times!
#		time_samples = self.manager.std_targets * time_samples + self.manager.mean_targets
		time_samples = time_samples * (self.manager.max_targets - self.manager.min_targets) + self.manager.min_targets

		
# get mean dissociation times
		mean_times = np.mean(time_samples, axis = 0)
		std_times  = np.std(time_samples, axis = 0)

		# now return everything
		return np.squeeze(mean_times), np.squeeze(std_times), time_samples


#======================================================================================