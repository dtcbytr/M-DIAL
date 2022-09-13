import os
import time
import cv2 as cv
import pygame as pg

def get_environment(args):
	env_name = args.environment

	if env_name == 'gym_blind_group_up':
		import GymBlindGroupUp
		return GymBlindGroupUp.GymBGU(args.n_agents, args.map_size)
	elif env_name == 'gym_navigation':
		import GymNavigation
		return GymNavigation.GymNav(args.n_agents, args.map_size)
	elif env_name == 'gym_pursuit':
		import GymPursuit
		return GymPursuit.GymPursuit(args.n_agents, args.obs_radius, args.n_prey, args.map_size)
	elif env_name == 'gym_traffic':
		import GymTraffic
		return GymTraffic.GymTraffic(args.intersections, args.n_agents, args.road_size, args.frequency)
	elif env_name == 'pong':
		import Pong
		return Pong.Pong()
	else:
		print('Undefined environment ' + env_name)
		quit()

def play_random_episode(environment, args):
	environment.reset()
	timestep = 0
	if args.render:
		environment.render()
		time.sleep(0.1)
	if args.log_dir != None:
		pg.image.save(environment.screen, os.path.join(args.log_dir, str(timestep) + '.png'))
	done = False
	while not done:
		timestep += 1
		action_space = environment.action_space
		sample_actions = [_.sample() for _ in action_space]

		_, _, done, _ = environment.step(sample_actions)
		if args.render:
			environment.render()
			time.sleep(0.1)
		if args.log_dir != None:
			pg.image.save(environment.screen, os.path.join(args.log_dir, str(timestep) + '.png'))

if __name__ == '__main__':
	from argparse import ArgumentParser
	argparse = ArgumentParser()
	argparse.add_argument('--log_dir', type = str)

	argparse.add_argument('--render', action = "store_true")

	argparse.add_argument('--environment', type = str)
	argparse.add_argument('--n_agents', type = int)
	argparse.add_argument('--map_size', type = int)
	argparse.add_argument('--obs_radius', type = int)
	argparse.add_argument('--n_prey', type = int)
	argparse.add_argument('--intersections', type = int)
	argparse.add_argument('--road_size', type = int)
	argparse.add_argument('--frequency', type = float)

	args = argparse.parse_args()

	# create environment
	environment = get_environment(args)
	play_random_episode(environment, args)
