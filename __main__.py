import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import networks
from train import train, test_episode
from utils import *
import dill
import numpy as np

def get_environment(args):
	env_name = args.environment

	import simulator

	if env_name == 'gym_blind_group_up':
		import simulator.GymBlindGroupUp
		return simulator.GymBlindGroupUp.GymBGU(args.n_agents, args.map_size)
	elif env_name == 'gym_navigation':
		import simulator.GymNavigation
		return simulator.GymNavigation.GymNav(args.n_agents, args.map_size)
	elif env_name == 'gym_pursuit':
		import simulator.GymPursuit
		return simulator.GymPursuit.GymPursuit(args.n_agents, args.obs_radius, args.n_prey, args.map_size)
	elif env_name == 'gym_traffic':
		import simulator.GymTraffic
		return simulator.GymTraffic.GymTraffic(args.intersections, args.n_agents, args.road_size, args.frequency)
	elif env_name == 'pong':
		import simulator.Pong
		return simulator.Pong.Pong()
	else:
		print('Undefined environment ' + env_name)
		quit()

def get_optimizer(args):
	optimizer_name = args.optimizer

	if optimizer_name == 'sgd':
		return tf.keras.optimizers.SGD(args.learning_rate)
	elif optimizer_name == 'adam':
		return tf.keras.optimizers.Adam(args.learning_rate)
	elif optimizer_name == 'rms_prop':
		return tf.keras.optimizers.RMSprop(args.learning_rate)
	else:
		print('Unknown optimizer ' + optimizer_name)
		quit()

if __name__ == '__main__':
	from argparse import ArgumentParser
	argparse = ArgumentParser()
	argparse.add_argument('--log_dir', action = readable_dir)
	argparse.add_argument('--learning_rate', type = float)

	argparse.add_argument('--render', action = "store_true")

	argparse.add_argument('--environment', type = str)
	argparse.add_argument('--n_agents', type = int)
	argparse.add_argument('--map_size', type = int)
	argparse.add_argument('--obs_radius', type = int)
	argparse.add_argument('--n_prey', type = int)
	argparse.add_argument('--intersections', type = int)
	argparse.add_argument('--road_size', type = int)
	argparse.add_argument('--frequency', type = float)

	argparse.add_argument('--epsilon', type = float)
	argparse.add_argument('--optimizer', type = str)
	argparse.add_argument('--epochs', type = int)
	argparse.add_argument('--episodes_per_epoch', type = int)
	argparse.add_argument('--tests_per_epoch', type = int)
	argparse.add_argument('--gamma', type = float)
	argparse.add_argument('--agent_message_output_dims', type = int)
	argparse.add_argument('--broadcast_message_dims', type = int)

	argparse.add_argument('--checkpoint_dir', action = readable_dir)

	argparse.add_argument('--test', type = int)
	argparse.add_argument('--checkpoint_mediator_path', action = readable_file)
	argparse.add_argument('--checkpoint_agent_path', action = readable_file)

	argparse.add_argument('--no_comm', action = 'store_true')

	argparse.add_argument('--use_lstm_layers', action = 'store_true')
	argparse.add_argument('--dial', action = 'store_true')

	args = argparse.parse_args()

	# create environment
	environment = get_environment(args)

	if args.test != None:

		agent_network = networks.Agent(n_actions = environment.action_space[0].n, message_output_dims = args.agent_message_output_dims, use_lstm_layers = args.use_lstm_layers)
		if args.dial:
			mediator_network = networks.JDIAL()
		else:
			mediator_network = networks.Mediator(message_output_dims = args.broadcast_message_dims, no_comm = args.no_comm)

		with open(args.checkpoint_agent_path, 'rb+') as file:
			agent_network_params = dill.load(file).get_weights()
		with open(args.checkpoint_mediator_path, 'rb+') as file:
			mediator_network_params = dill.load(file).get_weights()

		# Initializing run
		test_episode(agent_network, mediator_network, environment, args)

		agent_network.set_weights(agent_network_params)
		mediator_network.set_weights(mediator_network_params)
		reward_log = []
		hit_log = []
		args.broadcast_message_dims = mediator_network.message_output_dims
		for episode in range(args.test):
			reward, hits = test_episode(agent_network, mediator_network, environment, args, return_hits = True)
			reward_log.append(reward)
			hit_log.append(hits)
		print(np.mean(hit_log))
		quit()

	if args.dial:
		args.broadcast_message_dims = args.n_agents * args.agent_message_output_dims

	os.system('rm ' + os.path.join(args.log_dir, '*'))
	print('Cleared previous logs ' + os.path.join(args.log_dir, '*'))


	import json
	json.dump(vars(args), open(os.path.join(args.checkpoint_dir, 'info.json'), 'w'))

	# get networks and optimizer
	optimizer = get_optimizer(args)

	agent_network = networks.Agent(n_actions = environment.action_space[0].n, message_output_dims = args.agent_message_output_dims, use_lstm_layers = args.use_lstm_layers)
	if args.dial:
		mediator_network = networks.JDIAL()
		target_mediator_network = networks.JDIAL()
	else:
		mediator_network = networks.Mediator(message_output_dims = args.broadcast_message_dims, no_comm = args.no_comm)
		target_mediator_network = networks.Mediator(message_output_dims = args.broadcast_message_dims, no_comm = args.no_comm)
	target_agent_network = networks.Agent(n_actions = environment.action_space[0].n, message_output_dims = args.agent_message_output_dims, use_lstm_layers = args.use_lstm_layers)


	train(agent_network, target_agent_network, mediator_network, target_mediator_network, environment, optimizer, args)
