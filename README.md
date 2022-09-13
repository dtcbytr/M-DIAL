# Installation
## Create a virtualenv
virtualenv env
## Activate virtual environment
source ./env/bin/activate
## Install requirements
pip3 install -r requirements.txt
# Run the following
python3 coordination/ --log_dir ./coordination/bin/logs --learning_rate 0.0005 --environment gym_blind_group_up --n_agents 6 --map_size 5 --epsilon 0.05 --optimizer rms_prop --epochs 50000 --episodes_per_epoch 100 --tests_per_epoch 20 --gamma 1.0 --agent_message_output_dims 2 --broadcast_message_dims 4
