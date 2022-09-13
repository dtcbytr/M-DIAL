import os
import itertools

# These are used together - zipped
map_size = [8, 15]
environment = ['gym_blind_group_up', 'gym_navigation'] #, 'gym_traffic']
n_agents = [4, 2]

# I HAVE SWITCHED OFF RENDERING FOR ALL

use_lstm = ['--use_lstm', '']#, '--use_lstm']

log_dir = './bin/m_dial/logs/'
checkpoint_dir = './bin/m_dial/ckpt/'
no_comm = '--no_comm'
dial = '--dial'

# These are crossed
agent_message_output_dims = [4,8] # 2,4,8
broadcast_message_dims = [8,16] # 4,8, 16

command = 'python3 ./__main__.py --learning_rate 0.00001 --log_dir {log_dir} --environment {environment} --n_agents {n_agents} --map_size {map_size} --epsilon 0.05 --optimizer adam --epochs 50000 --episodes_per_epoch 100 --tests_per_epoch 20 --gamma 0.95 --agent_message_output_dims {agent_message_output_dims} --broadcast_message_dims {broadcast_message_dims} {use_lstm} --obs_radius 5 --checkpoint_dir {checkpoint_dir} '

counter = 0

for environment_val, map_size_val, n_agents_val in zip(environment, map_size, n_agents): # iterates over environments
    for lstm_status in use_lstm: # iterates over whether to use LSTM layers or not: for agents
        for agent_message_output_dims_val, broadcast_message_dims_val in itertools.product(agent_message_output_dims, broadcast_message_dims):

            fork_log_dir = os.path.join(log_dir, str(counter))
            os.makedirs(fork_log_dir)
            checkpoint_directory = os.path.join(checkpoint_dir, str(counter))
            os.makedirs(checkpoint_directory)
            print(command.format(log_dir = fork_log_dir, environment = environment_val, n_agents = n_agents_val, map_size = map_size_val, agent_message_output_dims = agent_message_output_dims_val, broadcast_message_dims = broadcast_message_dims_val, use_lstm = lstm_status, checkpoint_dir = checkpoint_directory))
            # quit()
            fork_command = command.format(log_dir = fork_log_dir, environment = environment_val, n_agents = n_agents_val, map_size = map_size_val, agent_message_output_dims = agent_message_output_dims_val, broadcast_message_dims = broadcast_message_dims_val, use_lstm = lstm_status, checkpoint_dir = checkpoint_directory)
            # with open(os.path.join(log_dir, str(counter) + '_info'), 'w') as f:
            #     f.write(fork_command)
            print(os.path.join(fork_log_dir, 'info.txt'))
            os.system(fork_command + '&')
            counter += 1
