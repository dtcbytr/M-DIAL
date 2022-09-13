import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import networks
from train import test_episode, process_to_network_input_format
from utils import *
import dill
import numpy as np

def process_names(names, dir):
    count = len(names) // 2
    agents, mediators = [0 for i in range(count)], [0 for i in range(count)]
    for name in names:
        if 'info' in name:
            continue
        net, number = name.split('_')
        number = int(number)
        if net == 'AGENT':
            agents[number] = os.path.join(dir, name)
        else:
            mediators[number] = os.path.join(dir, name)
    return agents, mediators


from argparse import ArgumentParser
argparse = ArgumentParser()
argparse.add_argument('--checkpoint_path', action = readable_dir)
# argparse.add_argument('--checkpoint_mediator_path', action = readable_file)
argparse.add_argument('--log_dir')
args = argparse.parse_args()

agents, mediators = process_names(os.listdir(args.checkpoint_path), args.checkpoint_path)
# print(agents, mediators)
writer = tf.summary.create_file_writer(args.log_dir)
counter = 0

for checkpoint_agent_path, checkpoint_mediator_path in zip(agents, mediators):
    args.checkpoint_agent_path = checkpoint_agent_path
    args.checkpoint_mediator_path = checkpoint_mediator_path

    with open(args.checkpoint_agent_path, 'rb+') as file:
        agent_network_ = dill.load(file)
    with open(args.checkpoint_mediator_path, 'rb+') as file:
        mediator_network_ = dill.load(file)

    # Initializing run
    import simulator.Pong
    environment = simulator.Pong.Pong()
    args.broadcast_message_dims = mediator_network_.message_output_dims
    args.n_agents = 2
    args.agent_message_output_dims = agent_network_.message_output_dims
    args.no_comm = mediator_network_.no_comm
    args.use_lstm_layers = agent_network_.use_lstm_layers
    agent_network = networks.Agent(n_actions = environment.action_space[0].n, message_output_dims = args.agent_message_output_dims, use_lstm_layers = args.use_lstm_layers)
    mediator_network = networks.Mediator(message_output_dims = args.broadcast_message_dims, no_comm = args.no_comm)
    args.render = False
    args.test = 5
    #initialize agent and mediator nets:
    agent_network(process_to_network_input_format(tf.zeros([2, 4]), tf.zeros([1, args.broadcast_message_dims])))
    mediator_network(tf.zeros([1, args.agent_message_output_dims * 2]))

    agent_network.set_weights(agent_network_.get_weights())
    mediator_network.set_weights(mediator_network_.get_weights())

    # test_episode(agent_network, mediator_network, environment, args, return_hits = True)
    reward_log = []
    hit_log = []
    args.broadcast_message_dims = mediator_network.message_output_dims
    for episode in range(args.test):
        reward, hits = test_episode(agent_network, mediator_network, environment, args, return_hits = True)
        reward_log.append(reward)
        hit_log.append(hits)

    val = (sum(hit_log) / len(hit_log))
    print(val)
    with writer.as_default():
        tf.summary.scalar('Hits', val, step = counter)
    counter += 1
