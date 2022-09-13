import tensorflow as tf
from tensorflow.keras import layers
import dill as pickle

class ExpandDims(tf.keras.layers.Layer):
	def __init__(self):
		super(ExpandDims, self).__init__()

	def __call__(self, x):
		return tf.expand_dims(x, axis = 1)

class Agent(tf.keras.Model):
	def __init__(self, n_actions, message_output_dims, use_lstm_layers):
		super(Agent, self).__init__()
		self.n_actions = n_actions
		self.use_lstm_layers = use_lstm_layers
		self.message_output_dims = message_output_dims
		self._layers = []
		if not use_lstm_layers:
			self._layers = [
				layers.Dense(64, activation = tf.nn.leaky_relu),
				layers.Dense(128, activation = tf.nn.leaky_relu),
				layers.Dense(32, activation = tf.nn.leaky_relu),
				layers.Dense(n_actions + message_output_dims)
			]
		else:
			self._layers = [
				layers.Dense(64, activation = tf.nn.leaky_relu),
				ExpandDims(),
				layers.GRU(128, activation = tf.nn.tanh, stateful = True, return_sequences = True),
				layers.GRU(32, activation = tf.nn.tanh, stateful = True, return_sequences = True),
				layers.GRU(n_actions + message_output_dims, stateful = True, activation = tf.nn.tanh)
			]

	def __call__(self, input):
		for _layer in self._layers:
			input = _layer(input)
		return tf.split(input, [self.n_actions, self.message_output_dims], axis = 1)

	def save_model(self, file_name):
		with open(file_name, 'wb+') as file:
			pickle.dump(self, file)

class Mediator(tf.keras.Model):
	def __init__(self, message_output_dims, no_comm = False):
		super(Mediator, self).__init__()
		self.message_output_dims = message_output_dims
		self.no_comm = no_comm
		if(no_comm):
			return
		self._model = tf.keras.models.Sequential([
			layers.Dense(64, activation = tf.nn.leaky_relu),
			layers.Dense(128, activation = tf.nn.leaky_relu),
			layers.Dense(32, activation = tf.nn.leaky_relu),
			layers.Dense(message_output_dims)
		])

	def __call__(self, input):
		if(self.no_comm):
			return tf.zeros([1, 1], dtype = tf.float32)
		return self._model(input)

	def save_model(self, file_name):
		with open(file_name, 'wb+') as file:
			pickle.dump(self, file)

class JDIAL(tf.keras.Model):
	def __init__(self):
		super(JDIAL, self).__init__()

	def __call__(self, input):
		return input
	def save_model(self, path):
		return
