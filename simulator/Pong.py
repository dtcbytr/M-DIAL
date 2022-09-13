import numpy as np
from math import ceil, floor
import math
import cv2 as cv

class Space:
	def __init__(self, n):
		self.n = n
	def sample(self):
		return np.random.randint(0, self.n)

def check(x, x_max):
	return x >= 0 and x < x_max

class Ball:
	def __init__(self, x_init, y_init, vx_init, vy_init, x_max, y_max):
		self.x = x_init
		self.y = y_init
		self.speed = (vx_init ** 2 + vy_init ** 2) ** 0.5
		self.vx = vx_init / self.speed
		self.vy = vy_init / self.speed
		self.x_max = x_max
		self.y_max = y_max
		self.x_init = x_init
		self.y_init = y_init
		self.vx_init = self.vx
		self.vy_init = self.vy
		self.factor = 1.0012

	def move(self):
		cx, cy = (-1, -1)
		if(check(self.x + self.vx * self.speed, self.x_max)):
			self.x += self.vx * self.speed
			cx = 1
		if(check(self.y + self.vy * self.speed, self.y_max)):
			self.y += self.vy * self.speed
			cy = 1
		self.vx *= cx
		self.vy *= cy
		self.speed = min(self.speed * self.factor, 4.0)

		if(cx == -1 or cy == -1):
			self.move()

	def reset(self):
		self.x = self.x_init
		self.y = self.y_init
		self.speed = (self.vx_init ** 2 + self.vy_init ** 2) ** 0.5
		speed = self.speed
		self.vx = np.random.uniform(-speed * 3.0 / 5.0, speed * 3.0 / 5.0)
		self.vy = ((speed ** 2 - self.vx ** 2) ** 0.5) * (1.0 if np.random.uniform() < 0.5 else -1.0)


class Paddle:
	def __init__(self, x, length, x_max):
		self.x = x
		self.x_init = x
		self.x_max = x_max
		self.length = length

	def reset(self):
		self.x = self.x_init

	def move(self, dx):
		if(check(self.x + dx, self.x_max) and check(self.x + dx + self.length - 1, self.x_max)):
			self.x += dx

class Dummy:
	def __init__(self):
		self.n = 3

class Pong:
	def __init__(self, window_name = 'Pong', x_init = 50, y_init = 50, vx_init = 0.3, vy_init = 0.8, x_paddle_left = 45, x_paddle_right = 45):
		self.window_name = window_name
		self.first_render = True
		self.hits = 0
		self.action_space = [Space(3), Space(3)]
		self.window_x = 100
		self.window_y = 100
		self.paddle_length = 15
		self.paddle_width = 1
		self.left_paddle_position = 5
		self.right_paddle_position = self.window_y - self.left_paddle_position - 1
		self.ball = Ball(x_init, y_init, vx_init, vy_init, self.window_x, self.window_y)
		self.paddle_left = Paddle(x_paddle_left, self.paddle_length, self.window_x)
		self.paddle_right = Paddle(x_paddle_right, self.paddle_length, self.window_x)
		self.x_init, self.y_init = (x_init, y_init)
		self.vx_init, self.vy_init = (vx_init, vy_init)
		self.x_paddle_left, self.x_paddle_right = x_paddle_left, x_paddle_right
		self.done = False

	def reset(self):
		self.hits = 0
		self.done = False
		self.ball.reset()
		self.paddle_left.reset()
		self.paddle_right.reset()
		return np.asarray([self.get_left_observation(), self.get_right_observation()], np.float32), None

	def get_screen(self):
		screen = np.zeros([self.window_x, self.window_y])
		cv.line(screen, pt1 = (self.left_paddle_position, self.paddle_left.x), pt2 = (self.left_paddle_position, self.paddle_length + self.paddle_left.x), color = 1.0, lineType = 0, thickness = self.paddle_width)
		cv.line(screen, pt1 = (self.right_paddle_position, self.paddle_right.x), pt2 = (self.right_paddle_position, self.paddle_length + self.paddle_right.x), color = 1.0, lineType = 0, thickness = self.paddle_width)
		if(self.ball.x - floor(self.ball.x) > 0.5):
			c_x = ceil(self.ball.x)
		else:
			c_x = floor(self.ball.x) - 1
		if(self.ball.y - floor(self.ball.y) > 0.5):
			c_y = ceil(self.ball.y)
		else:
			c_y = floor(self.ball.y) - 1
		x, y = (floor(self.ball.x), floor(self.ball.y))
		points = [(c_y, c_x), (c_y, x), (y, c_x), (y, x)]
		intensity_controls = [0.6, 0.6, 0.6, 1.0]
		for i in range(4):
			if check(points[i][1], self.window_y) and check(points[i][0], self.window_x):
				screen[points[i][1], points[i][0]] = intensity_controls[i]

		return screen
	@property
	def screen(self):
		import pygame
		output = self.get_screen()
		output = 255 * output / output.max()
		return pygame.surfarray.make_surface(output)

	def get_state(self):	# returns a numpy array of shape [4]
		return np.asarray([self.ball.x, self.ball.y, self.paddle_left.x, self.paddle_right.x])

	def get_left_observation(self, view_l = 50):
		visible = [1.0, self.ball.x, self.ball.y] if self.ball.y < view_l else [0.0, 0.0, 0.0]
		return np.asarray(visible + [self.paddle_left.x])

	def get_right_observation(self, view_r = 50):
		visible = [1.0, self.ball.x, self.window_y - 1 - self.ball.y] if self.ball.y >= (self.window_y - view_r) else [0.0, 0.0, 0.0]
		return np.asarray(visible + [self.paddle_right.x])

		self.hits = 0
		self.done = False
		self.ball.reset()
		self.paddle_left.reset()
		self.paddle_right.reset()
	def move(self, l_move, r_move):
		# move can be 0, 1 or 2
		# 0 -> no_op
		# 1 -> down
		# 2 -> up
		if(l_move == 2):
			l_move = -1
		if(r_move == 2):
			r_move = -1
		self.paddle_left.move(l_move)
		self.paddle_right.move(r_move)
		# check for paddle and move
		# reset if something bad
		if((self.paddle_right.x <= self.ball.x and self.paddle_right.x + self.paddle_right.length - 1 >= self.ball.x and self.ball.y + self.ball.speed * self.ball.vy >= self.right_paddle_position)):
			#successfully blocked
			theta = (math.pi * 3 / 12) * ((self.ball.x - self.paddle_right.x) * 2.0 / self.paddle_right.length - 1.0)
			self.ball.vx, self.ball.vy = (math.sin(theta), -math.cos(theta))
			self.ball.move()
			self.hits += 1
			return 0.0, 0.0, False

		elif((self.paddle_left.x <= self.ball.x and self.paddle_left.x + self.paddle_left.length - 1 >= self.ball.x and self.ball.y + self.ball.speed * self.ball.vy <= self.left_paddle_position)):
			theta = (math.pi * 3 / 12) * ((self.ball.x - self.paddle_left.x) * 2.0 / self.paddle_left.length - 1.0)
			self.ball.vx, self.ball.vy = (math.sin(theta), math.cos(theta))
			self.ball.move()
			self.hits += 1
			return 0.0, 0.0, False


		elif((self.paddle_left.x > self.ball.x or self.paddle_left.x + self.paddle_left.length - 1 < self.ball.x) and self.ball.y + self.ball.speed * self.ball.vy <= self.left_paddle_position):
			#missed
			self.ball.move()
			self.done = True
			return -1.0, -1.0, True
		elif((self.paddle_right.x > self.ball.x or self.paddle_right.x + self.paddle_right.length - 1 < self.ball.x) and self.ball.y + self.ball.speed * self.ball.vy >= self.right_paddle_position):
			self.ball.move()
			self.done = True
			return -1.0, -1.0, True

		else:
			self.ball.move()
			return 0.0, 0.0, False

	def step(self, actions):
		l_move, r_move = actions
		assert self.done == False, "Environment should be reset for further use"
		import random
		done = False
		reward_l, reward_r = 0.0, 0.0
		for _ in range(random.randint(2, 4)):
			reward_l, reward_r, done = self.move(l_move, r_move)
			if done:
				break
		return np.asarray([self.get_left_observation(), self.get_right_observation()]), np.asarray([reward_l, reward_r]), done, None

	def render(self):
		if self.first_render:
			self.first_render = False
			cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
			cv.resizeWindow(self.window_name, 100, 100)

		cv.imshow(self.window_name, self.get_screen())
		cv.waitKey(1)
