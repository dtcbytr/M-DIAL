import tensorflow as tf
from tqdm import tqdm
from copy import deepcopy
import pickle
import numpy as np
import os

def process_to_network_input_format(observations, broadcast_message):
    broadcast_message = tf.tile(broadcast_message, [len(observations), 1])
    observations = tf.cast(tf.convert_to_tensor(observations), tf.float32)

    return tf.concat([observations, broadcast_message], axis = -1)


def sample_actions(q_values, epsilon):
    actions = []
    for i in range(q_values.shape[0]):
        q_value = q_values[i]
        if np.random.uniform() < epsilon:
            actions.append(np.random.randint(0, q_values.shape[-1], dtype = np.int))
        else:
            actions.append(np.argmax(q_value))
    return actions

def test_episode(agent_network, mediator_network, environment, args, return_hits = False):
    try:
        agent_network.reset_states()
    except:
        pass
    observations = environment.reset()[0]
    done = False

    broadcast_message = tf.zeros(shape = [1, args.broadcast_message_dims])

    total_reward = np.zeros([args.n_agents])

    while not done:
        agent_network_input = process_to_network_input_format(observations, broadcast_message)
        q_values, messages = agent_network(agent_network_input)

        mediator_network_input = tf.reshape(messages, [1, -1])
        broadcast_message = mediator_network(mediator_network_input)

        actions = sample_actions(q_values = q_values, epsilon = -1.0)
        if args.render:
        	environment.render()
        observations, reward, done, info = environment.step(actions)

        total_reward += reward
    if return_hits:
        return np.sum(total_reward), environment.hits
    return np.sum(total_reward)

def training_episode(agent_network, target_agent_network, mediator_network, target_mediator_network, environment, args):
    try:
        agent_network.reset_states()
    except:
        pass
    observations = environment.reset()[0]
    done = False

    broadcast_message = tf.zeros(shape = [1, args.broadcast_message_dims])
    target_broadcast_message = tf.zeros(shape = [1, args.broadcast_message_dims])

    target_agent_network_input = process_to_network_input_format(observations, target_broadcast_message)
    _, target_messages = target_agent_network(target_agent_network_input)

    target_mediator_network_input = tf.reshape(target_messages, [1, -1])
    target_broadcast_message = target_mediator_network(target_mediator_network_input)

    total_reward = np.zeros([args.n_agents])
    loss = 0.0
    time_steps = 0

    while not done:
        time_steps += 1
        agent_network_input = process_to_network_input_format(observations, broadcast_message)
        q_values, messages = agent_network(agent_network_input)

        mediator_network_input = tf.reshape(messages, [1, -1])
        broadcast_message = mediator_network(mediator_network_input)

        actions = sample_actions(q_values = q_values, epsilon = args.epsilon)

        if args.render:
        	environment.render()
        observations, reward, done, info = environment.step(actions)
        reward = tf.convert_to_tensor([float(i) for i in reward])

        total_reward += reward

        if done:
            index = tf.convert_to_tensor(list(enumerate(actions)))
            loss += tf.reduce_mean(tf.square(tf.convert_to_tensor(reward) - tf.gather_nd(q_values, index)))

        else:
            target_agent_network_input = process_to_network_input_format(observations, target_broadcast_message)
            target_q_values, target_messages = target_agent_network(target_agent_network_input)

            target_mediator_network_input = tf.reshape(target_messages, [1, -1])
            target_broadcast_message = target_mediator_network(target_mediator_network_input)

            index = tf.convert_to_tensor(list(enumerate(actions)))
            loss += tf.reduce_mean(tf.square(tf.convert_to_tensor(reward) + args.gamma * tf.reduce_max(target_q_values, axis = -1) - tf.gather_nd(q_values, index)))

    return loss, total_reward


def train(agent_network, target_agent_network, mediator_network, target_mediator_network, environment, optimizer, args):
    summary_writer = tf.summary.create_file_writer(args.log_dir)
    # one test to initialize
    test_episode(agent_network, mediator_network, environment, args)

    for epoch in range(args.epochs):
        for episode in tqdm(range(args.episodes_per_epoch), desc = 'Epoch {:6d}'.format(epoch)):
            with tf.GradientTape() as tape:
                loss, reward = training_episode(agent_network, target_agent_network, mediator_network, target_mediator_network, environment, args)

            gradients = tape.gradient(loss, agent_network.trainable_variables + mediator_network.trainable_variables)
            optimizer.apply_gradients(zip(gradients, agent_network.trainable_variables + mediator_network.trainable_variables))

        # update target networks
        target_mediator_network.set_weights(mediator_network.get_weights())
        target_agent_network.set_weights(agent_network.get_weights())

        # log performance
        reward_list = np.asarray([test_episode(agent_network, mediator_network, environment, args) for _ in range(args.tests_per_epoch)])
        with summary_writer.as_default():
            tf.summary.scalar('average-reward', np.mean(reward_list), epoch)

        # save checkpoint
        if args.checkpoint_dir != None:
            agent_network.save_model(os.path.join(args.checkpoint_dir, 'AGENT_'+ str(epoch)))
            mediator_network.save_model(os.path.join(args.checkpoint_dir, 'MEDIATOR_' + str(epoch)))
